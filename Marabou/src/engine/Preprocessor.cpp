/*********************                                                        */
/*! \file Preprocessor.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Derek Huang, Shantanu Thakoor, Haoze Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/


#include "Preprocessor.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "InfeasibleQueryException.h"
#include "InputQuery.h"
#include "MStringf.h"
#include "Map.h"
#include "MarabouError.h"
#include "Options.h"
#include "PiecewiseLinearFunctionType.h"
#include "Statistics.h"
#include "Tightening.h"
#ifdef _WIN32
#undef INFINITE
#endif

Preprocessor::Preprocessor()
    : _preprocessed( nullptr )
    , _statistics( NULL )
    , _lowerBounds( NULL )
    , _upperBounds( NULL )
{
}

Preprocessor::~Preprocessor()
{
    freeMemoryIfNeeded();
}

void Preprocessor::freeMemoryIfNeeded()
{
    if ( _lowerBounds != NULL )
    {
        delete[] _lowerBounds;
        _lowerBounds = NULL;
    }
    if ( _upperBounds != NULL )
    {
        delete[] _upperBounds;
        _upperBounds = NULL;
    }
}

std::unique_ptr<InputQuery> Preprocessor::preprocess( const InputQuery &query,
                                                      bool attemptVariableElimination )
{
    _preprocessed = std::unique_ptr<InputQuery>( new InputQuery( query ) );

    /*
      Next, make sure all equations are of type EQUALITY. If not, turn them
      into one.
    */
    makeAllEquationsEqualities();

    /*
      Attempt to construct a network level reasoner
    */
    List<Equation> unhandledEquations;
    Set<unsigned> varsInUnhandledConstraints;
    _preprocessed->constructNetworkLevelReasoner( unhandledEquations, varsInUnhandledConstraints );

    /*
      Merge consecutive WS layers
    */
    if ( !Options::get()->getBool( Options::DO_NOT_MERGE_CONSECUTIVE_WEIGHTED_SUM_LAYERS ) )
    {
        _preprocessed->mergeConsecutiveWeightedSumLayers(
            unhandledEquations, varsInUnhandledConstraints, _unusedSymbolicallyFixedVariables );
    }

    removeRedundantAddendsInAllEquations();

    /*
      Transform the piecewise linear constraints if needed so that the case
      splits can all be represented as bounds over existing variables.
    */
    transformConstraintsIfNeeded();

    /*
      Collect input and output variables
    */
    for ( const auto &var : _preprocessed->getInputVariables() )
        _uneliminableVariables.insert( var );
    for ( const auto &var : _preprocessed->getOutputVariables() )
        _uneliminableVariables.insert( var );
    for ( const auto &constraint : _preprocessed->getPiecewiseLinearConstraints() )
        if ( !constraint->supportVariableElimination() )
            for ( const auto &var : constraint->getParticipatingVariables() )
                _uneliminableVariables.insert( var );
    for ( const auto &constraint : _preprocessed->getNonlinearConstraints() )
    {
        if ( !constraint->supportVariableElimination() )
            for ( const auto &var : constraint->getParticipatingVariables() )
                _uneliminableVariables.insert( var );
    }

    /*
      Set any missing bounds
    */
    setMissingBoundsToInfinity();

    /*
      Store the bounds locally for more efficient access.
    */
    _lowerBounds = new double[_preprocessed->getNumberOfVariables()];
    _upperBounds = new double[_preprocessed->getNumberOfVariables()];

    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        _lowerBounds[i] = _preprocessed->getLowerBound( i );
        _upperBounds[i] = _preprocessed->getUpperBound( i );
    }

    /*
      Do the preprocessing steps:

      Until saturation:
        1. Tighten bounds using equations
        2. Tighten bounds using pl constraints

      Then, eliminate fixed variables.
    */
    unsigned tighteningRound = 0;
    bool continueTightening = true;
    while ( continueTightening &&
            tighteningRound++ < GlobalConfiguration::PREPROCESSSING_MAX_TIGHTEING_ROUND )
    {
        DEBUG( {
            for ( const auto &equation : _preprocessed->getEquations() )
                ASSERT( !equation.containsRedundantAddends() );
        } );
        continueTightening = processEquations();
        continueTightening = processConstraints() || continueTightening;
        if ( attemptVariableElimination )
            continueTightening = processIdenticalVariables() || continueTightening;

        if ( _statistics )
            _statistics->incUnsignedAttribute( Statistics::PP_NUM_TIGHTENING_ITERATIONS );
    }

    collectFixedValues();
    separateMergedAndFixed();

    if ( attemptVariableElimination )
        eliminateVariables();

    /*
      Update the bounds.
    */
    _preprocessed->clearBounds();
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        _preprocessed->setLowerBound( i, getLowerBound( i ) );
        _preprocessed->setUpperBound( i, getUpperBound( i ) );
    }

    ASSERT( _preprocessed->getLowerBounds().size() == _preprocessed->getNumberOfVariables() );
    ASSERT( _preprocessed->getUpperBounds().size() == _preprocessed->getNumberOfVariables() );


    String networkFilePath = Options::get()->getString( Options::INPUT_FILE_PATH );




    return std::move( _preprocessed );
}

void Preprocessor::separateMergedAndFixed()
{
    Map<unsigned, double> noLongerMerged;

    for ( const auto &merged : _mergedVariables )
    {
        // In case of a chained merging, go all the way to the final target
        unsigned finalMergeTarget = merged.second;
        while ( _mergedVariables.exists( finalMergeTarget ) )
            finalMergeTarget = _mergedVariables[finalMergeTarget];

        // Is the merge target fixed?
        if ( _fixedVariables.exists( finalMergeTarget ) )
            noLongerMerged[merged.first] = _fixedVariables[finalMergeTarget];
    }

    // We have collected all the merged variables that should actually be fixed
    for ( const auto &merged : noLongerMerged )
    {
        _mergedVariables.erase( merged.first );
        _fixedVariables[merged.first] = merged.second;
    }

    DEBUG( {
        // After this operation, the merged and fixed variable sets are disjoint
        for ( const auto &fixed : _fixedVariables )
            ASSERT( !_mergedVariables.exists( fixed.first ) );
    } );
}

void Preprocessor::transformConstraintsIfNeeded()
{
    for ( auto &plConstraint : _preprocessed->getPiecewiseLinearConstraints() )
        plConstraint->transformToUseAuxVariables( *_preprocessed );
}

void Preprocessor::removeRedundantAddendsInAllEquations()
{
    for ( auto &equation : _preprocessed->getEquations() )
        equation.removeRedundantAddends();
}

void Preprocessor::makeAllEquationsEqualities()
{
    for ( auto &equation : _preprocessed->getEquations() )
    {
        if ( equation._type == Equation::EQ )
            continue;

        unsigned auxVariable = _preprocessed->getNumberOfVariables();
        _preprocessed->setNumberOfVariables( auxVariable + 1 );

        // Auxiliary variables are always added with coefficient 1
        if ( equation._type == Equation::GE )
            _preprocessed->setUpperBound( auxVariable, 0 );
        else
            _preprocessed->setLowerBound( auxVariable, 0 );

        equation._type = Equation::EQ;

        equation.addAddend( 1, auxVariable );
    }
}

bool Preprocessor::processEquations()
{
    enum {
        ZERO = 0,
        POSITIVE = 1,
        NEGATIVE = 2,
        INFINITE = 3,
    };

    bool tighterBoundFound = false;

    List<Equation> &equations( _preprocessed->getEquations() );
    List<Equation>::iterator equation = equations.begin();

    while ( equation != equations.end() )
    {
        // The equation is of the form sum (ci * xi) - b ? 0
        Equation::EquationType type = equation->_type;

        unsigned maxVar = equation->_addends.begin()->_variable;
        for ( const auto &addend : equation->_addends )
        {
            if ( addend._variable > maxVar )
                maxVar = addend._variable;
        }

        ++maxVar;

        double *ciTimesLb = new double[maxVar];
        double *ciTimesUb = new double[maxVar];
        char *ciSign = new char[maxVar];

        Set<unsigned> excludedFromLB;
        Set<unsigned> excludedFromUB;

        unsigned xi;
        double xiLB;
        double xiUB;
        double ci;
        double lowerBound;
        double upperBound;
        bool validLb;
        bool validUb;

        // The first goal is to compute the LB and UB of: sum (ci * xi) - b
        // For this we first identify unbounded variables
        double auxLb = -equation->_scalar;
        double auxUb = -equation->_scalar;
        for ( const auto &addend : equation->_addends )
        {
            ci = addend._coefficient;
            xi = addend._variable;

            if ( FloatUtils::isZero( ci ) )
            {
                ciSign[xi] = ZERO;
                ciTimesLb[xi] = 0;
                ciTimesUb[xi] = 0;
                continue;
            }

            ciSign[xi] = ci > 0 ? POSITIVE : NEGATIVE;

            xiLB = getLowerBound( xi );
            xiUB = getUpperBound( xi );

            if ( FloatUtils::isFinite( xiLB ) )
            {
                ciTimesLb[xi] = ci * xiLB;
                if ( ciSign[xi] == POSITIVE )
                    auxLb += ciTimesLb[xi];
                else
                    auxUb += ciTimesLb[xi];
            }
            else
            {
                if ( ci > 0 )
                    excludedFromLB.insert( xi );
                else
                    excludedFromUB.insert( xi );
            }

            if ( FloatUtils::isFinite( xiUB ) )
            {
                ciTimesUb[xi] = ci * xiUB;
                if ( ciSign[xi] == POSITIVE )
                    auxUb += ciTimesUb[xi];
                else
                    auxLb += ciTimesUb[xi];
            }
            else
            {
                if ( ci > 0 )
                    excludedFromUB.insert( xi );
                else
                    excludedFromLB.insert( xi );
            }
        }

        // Now, go over each addend in sum (ci * xi) - b ? 0, and see what can be done
        for ( const auto &addend : equation->_addends )
        {
            ci = addend._coefficient;
            xi = addend._variable;

            // If ci = 0, nothing to do.
            if ( ciSign[xi] == ZERO )
                continue;

            /*
              The expression for xi is:

                   xi ? ( -1/ci ) * ( sum_{j\neqi} ( cj * xj ) - b )

              We use the previously computed auxLb and auxUb and adjust them because
              xi is removed from the sum. We also need to pay attention to the sign of ci,
              and to the presence of infinite bounds.

              Assuming "?" stands for equality, we can compute a LB if:
                1. ci is negative, and no vars except xi were excluded from the auxLb
                2. ci is positive, and no vars except xi were excluded from the auxUb

              And vice-versa for UB.

              In case "?" is GE or LE, only one direction can be computed.
            */
            if ( ciSign[xi] == NEGATIVE )
            {
                validLb = ( ( type == Equation::LE ) || ( type == Equation::EQ ) ) &&
                          ( excludedFromLB.empty() ||
                            ( excludedFromLB.size() == 1 && excludedFromLB.exists( xi ) ) );
                validUb = ( ( type == Equation::GE ) || ( type == Equation::EQ ) ) &&
                          ( excludedFromUB.empty() ||
                            ( excludedFromUB.size() == 1 && excludedFromUB.exists( xi ) ) );
            }
            else
            {
                validLb = ( ( type == Equation::GE ) || ( type == Equation::EQ ) ) &&
                          ( excludedFromUB.empty() ||
                            ( excludedFromUB.size() == 1 && excludedFromUB.exists( xi ) ) );
                validUb = ( ( type == Equation::LE ) || ( type == Equation::EQ ) ) &&
                          ( excludedFromLB.empty() ||
                            ( excludedFromLB.size() == 1 && excludedFromLB.exists( xi ) ) );
            }

            // Now compute the actual bounds and see if they are tighter
            double epsilon = Options::get()->getFloat( Options::PREPROCESSOR_BOUND_TOLERANCE );

            if ( validLb )
            {
                if ( ciSign[xi] == NEGATIVE )
                {
                    lowerBound = auxLb;
                    if ( !excludedFromLB.exists( xi ) )
                        lowerBound -= ciTimesUb[xi];
                }
                else
                {
                    lowerBound = auxUb;
                    if ( !excludedFromUB.exists( xi ) )
                        lowerBound -= ciTimesUb[xi];
                }

                lowerBound /= -ci;

                if ( FloatUtils::gt( lowerBound, getLowerBound( xi ), epsilon ) )
                {
                    tighterBoundFound = true;
                    setLowerBound( xi, lowerBound );
                }
            }

            if ( validUb )
            {
                if ( ciSign[xi] == NEGATIVE )
                {
                    upperBound = auxUb;
                    if ( !excludedFromUB.exists( xi ) )
                        upperBound -= ciTimesLb[xi];
                }
                else
                {
                    upperBound = auxLb;
                    if ( !excludedFromLB.exists( xi ) )
                        upperBound -= ciTimesLb[xi];
                }

                upperBound /= -ci;

                if ( FloatUtils::lt( upperBound, getUpperBound( xi ), epsilon ) )
                {
                    tighterBoundFound = true;
                    setUpperBound( xi, upperBound );
                }
            }

            if ( FloatUtils::gt( getLowerBound( xi ),
                                 getUpperBound( xi ),
                                 GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
            {
                delete[] ciTimesLb;
                delete[] ciTimesUb;
                delete[] ciSign;

                throw InfeasibleQueryException();
            }
        }

        delete[] ciTimesLb;
        delete[] ciTimesUb;
        delete[] ciSign;

        /*
          Next, do another sweep over the equation.
          Look for almost-fixed variables and fix them, and remove the equation
          entirely if it has nothing left to contribute.
        */
        bool allFixed = true;
        for ( const auto &addend : equation->_addends )
        {
            unsigned var = addend._variable;
            double lb = getLowerBound( var );
            double ub = getUpperBound( var );

            if ( FloatUtils::areEqual(
                     lb, ub, GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
                setUpperBound( var, getLowerBound( var ) );
            else
                allFixed = false;
        }

        if ( !allFixed )
        {
            ++equation;
        }
        else
        {
            double sum = 0;
            for ( const auto &addend : equation->_addends )
                sum += addend._coefficient * getLowerBound( addend._variable );

            if ( FloatUtils::areDisequal(
                     sum,
                     equation->_scalar,
                     GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
            {
                throw InfeasibleQueryException();
            }
            equation = equations.erase( equation );
        }
    }

    return tighterBoundFound;
}

bool Preprocessor::processConstraints()
{
    bool tighterBoundFound = false;

    for ( auto &constraint : _preprocessed->getPiecewiseLinearConstraints() )
    {
        for ( unsigned variable : constraint->getParticipatingVariables() )
        {
            constraint->notifyLowerBound( variable, getLowerBound( variable ) );
            constraint->notifyUpperBound( variable, getUpperBound( variable ) );
        }

        List<Tightening> tightenings;
        constraint->getEntailedTightenings( tightenings );

        for ( const auto &tightening : tightenings )
        {
            if ( ( tightening._type == Tightening::LB ) &&
                 ( FloatUtils::gt( tightening._value, getLowerBound( tightening._variable ) ) ) )
            {
                tighterBoundFound = true;
                setLowerBound( tightening._variable, tightening._value );
            }

            else if ( ( tightening._type == Tightening::UB ) &&
                      ( FloatUtils::lt( tightening._value,
                                        getUpperBound( tightening._variable ) ) ) )
            {
                tighterBoundFound = true;
                setUpperBound( tightening._variable, tightening._value );
            }

            if ( FloatUtils::areEqual( getLowerBound( tightening._variable ),
                                       getUpperBound( tightening._variable ),
                                       GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
                setUpperBound( tightening._variable, getLowerBound( tightening._variable ) );

            if ( FloatUtils::gt( getLowerBound( tightening._variable ),
                                 getUpperBound( tightening._variable ),
                                 GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
            {
                throw InfeasibleQueryException();
            }
        }
    }

    for ( auto &constraint : _preprocessed->getNonlinearConstraints() )
    {
        for ( unsigned variable : constraint->getParticipatingVariables() )
        {
            constraint->notifyLowerBound( variable, getLowerBound( variable ) );
            constraint->notifyUpperBound( variable, getUpperBound( variable ) );
        }

        List<Tightening> tightenings;
        constraint->getEntailedTightenings( tightenings );

        for ( const auto &tightening : tightenings )
        {
            if ( ( tightening._type == Tightening::LB ) &&
                 ( FloatUtils::gt( tightening._value, getLowerBound( tightening._variable ) ) ) )
            {
                tighterBoundFound = true;
                setLowerBound( tightening._variable, tightening._value );
            }

            else if ( ( tightening._type == Tightening::UB ) &&
                      ( FloatUtils::lt( tightening._value,
                                        getUpperBound( tightening._variable ) ) ) )
            {
                tighterBoundFound = true;
                setUpperBound( tightening._variable, tightening._value );
            }

            if ( FloatUtils::areEqual( getLowerBound( tightening._variable ),
                                       getUpperBound( tightening._variable ),
                                       GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
                setUpperBound( tightening._variable, getLowerBound( tightening._variable ) );

            if ( FloatUtils::gt( getLowerBound( tightening._variable ),
                                 getUpperBound( tightening._variable ),
                                 GlobalConfiguration::PREPROCESSOR_ALMOST_FIXED_THRESHOLD ) )
            {
                throw InfeasibleQueryException();
            }
        }
    }

    return tighterBoundFound;
}

bool Preprocessor::processIdenticalVariables()
{
    List<Equation> &equations( _preprocessed->getEquations() );
    List<Equation>::iterator equation = equations.begin();

    bool found = false;
    while ( equation != equations.end() )
    {
        // We are only looking for equations of type c(v1 - v2) = 0
        if ( equation->_addends.size() != 2 || equation->_type != Equation::EQ )
        {
            ++equation;
            continue;
        }

        Equation::Addend term1 = equation->_addends.front();
        Equation::Addend term2 = equation->_addends.back();

        if ( FloatUtils::areDisequal( term1._coefficient, -term2._coefficient ) ||
             !FloatUtils::isZero( equation->_scalar ) )
        {
            ++equation;
            continue;
        }

        ASSERT( term1._variable != term2._variable );

        // The equation matches the pattern, extract the variables
        unsigned v1 = term1._variable;
        unsigned v2 = term2._variable;

        // Input and output variables should not be merged
        if ( _uneliminableVariables.exists( v1 ) || _uneliminableVariables.exists( v2 ) )
        {
            ++equation;
            continue;
        }

        // This equation can be removed
        found = true;

        double bestLowerBound =
            getLowerBound( v1 ) > getLowerBound( v2 ) ? getLowerBound( v1 ) : getLowerBound( v2 );

        double bestUpperBound =
            getUpperBound( v1 ) < getUpperBound( v2 ) ? getUpperBound( v1 ) : getUpperBound( v2 );

        equation = equations.erase( equation );

        setLowerBound( v2, bestLowerBound );
        setUpperBound( v2, bestUpperBound );

        _preprocessed->mergeIdenticalVariables( v1, v2 );

        _mergedVariables[v1] = v2;
    }

    return found;
}

void Preprocessor::collectFixedValues()
{
    // Compute all used variables:
    //   1. Variables that appear in equations
    //   2. Variables that participate in PL and nonlinear constraints
    //   3. Variables that have been merged (and hence, previously
    //      appeared in an equation)
    Set<unsigned> usedVariables;
    for ( const auto &equation : _preprocessed->getEquations() )
        usedVariables += equation.getParticipatingVariables();
    for ( const auto &constraint : _preprocessed->getPiecewiseLinearConstraints() )
    {
        for ( const auto &var : constraint->getParticipatingVariables() )
            usedVariables.insert( var );
    }
    for ( const auto &constraint : _preprocessed->getNonlinearConstraints() )
    {
        for ( const auto &var : constraint->getParticipatingVariables() )
            usedVariables.insert( var );
    }
    for ( const auto &merged : _mergedVariables )
        usedVariables.insert( merged.first );

    // Collect any variables with identical lower and upper bounds, or
    // which are unused
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( FloatUtils::areEqual( getLowerBound( i ), getUpperBound( i ) ) )
        {
            _fixedVariables[i] = getLowerBound( i );
        }
        else if ( !usedVariables.exists( i ) )
        {
            // If possible, choose a value that matches the debugging
            // solution. Otherwise, pick an arbitrary values. If the
            // bounds are infinite for this variable, set them
            // arbitrarily as well.
            if ( _preprocessed->_debuggingSolution.exists( i ) &&
                 _preprocessed->_debuggingSolution[i] >= getLowerBound( i ) &&
                 _preprocessed->_debuggingSolution[i] <= getUpperBound( i ) )
            {
                _fixedVariables[i] = _preprocessed->_debuggingSolution[i];
            }
            else
            {
                if ( FloatUtils::isFinite( getLowerBound( i ) ) )
                    _fixedVariables[i] = getLowerBound( i );
                else if ( FloatUtils::isFinite( getUpperBound( i ) ) )
                    _fixedVariables[i] = getUpperBound( i );
                else
                    _fixedVariables[i] = 0;
            }

            setLowerBound( i, _fixedVariables[i] );
            setUpperBound( i, _fixedVariables[i] );
        }
    }
}

void Preprocessor::eliminateVariables()
{
    // If there's nothing to eliminate, we just eliminate obsolete constraints.
    if ( _fixedVariables.empty() && _mergedVariables.empty() )
    {
        List<PiecewiseLinearConstraint *> &constraints(
            _preprocessed->getPiecewiseLinearConstraints() );
        List<PiecewiseLinearConstraint *>::iterator constraint = constraints.begin();
        while ( constraint != constraints.end() )
        {
            if ( ( *constraint )->constraintObsolete() )
            {
                if ( _statistics )
                    _statistics->incUnsignedAttribute( Statistics::PP_NUM_CONSTRAINTS_REMOVED );

                if ( _preprocessed->_networkLevelReasoner )
                    _preprocessed->_networkLevelReasoner->removeConstraintFromTopologicalOrder(
                        *constraint );
                delete *constraint;
                *constraint = NULL;
                constraint = constraints.erase( constraint );
            }
            else
                ++constraint;
        }

        List<NonlinearConstraint *> &nlConstraints( _preprocessed->getNonlinearConstraints() );
        List<NonlinearConstraint *>::iterator nlConstraint = nlConstraints.begin();
        while ( nlConstraint != nlConstraints.end() )
        {
            if ( ( *nlConstraint )->constraintObsolete() )
            {
                if ( _statistics )
                    _statistics->incUnsignedAttribute( Statistics::PP_NUM_CONSTRAINTS_REMOVED );

                delete *nlConstraint;
                *nlConstraint = NULL;
                nlConstraint = nlConstraints.erase( nlConstraint );
            }
            else
                ++nlConstraint;
        }
        return;
    }

    if ( _statistics )
        _statistics->setUnsignedAttribute( Statistics::PP_NUM_ELIMINATED_VARS,
                                           _fixedVariables.size() + _mergedVariables.size() );

    // Check and remove any fixed variables from the debugging solution
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( _fixedVariables.exists( i ) && _preprocessed->_debuggingSolution.exists( i ) )
        {
            if ( !FloatUtils::areEqual( _fixedVariables[i], _preprocessed->_debuggingSolution[i] ) )
                throw MarabouError( MarabouError::DEBUGGING_ERROR,
                                    Stringf( "Variable %u fixed to %.5lf, "
                                             "contradicts possible solution %.5lf",
                                             i,
                                             _fixedVariables[i],
                                             _preprocessed->_debuggingSolution[i] )
                                        .ascii() );

            _preprocessed->_debuggingSolution.erase( i );
        }
    }

    // Check and remove any merged variables from the debugging
    // solution
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( _mergedVariables.exists( i ) && _preprocessed->_debuggingSolution.exists( i ) )
        {
            double newVar = _mergedVariables[i];

            if ( _preprocessed->_debuggingSolution.exists( newVar ) )
            {
                if ( !FloatUtils::areEqual( _preprocessed->_debuggingSolution[i],
                                            _preprocessed->_debuggingSolution[newVar] ) )
                    throw MarabouError( MarabouError::DEBUGGING_ERROR,
                                        Stringf( "Variable %u fixed to %.5lf, "
                                                 "merged into %u which was fixed to %.5lf",
                                                 i,
                                                 _preprocessed->_debuggingSolution[i],
                                                 newVar,
                                                 _preprocessed->_debuggingSolution[newVar] )
                                            .ascii() );
            }
            else
            {
                _preprocessed->_debuggingSolution[newVar] = _preprocessed->_debuggingSolution[i];
            }

            _preprocessed->_debuggingSolution.erase( i );
        }
    }

    // Inform the NLR about eliminated varibales, unless they are
    // input/output variables
    if ( _preprocessed->_networkLevelReasoner )
    {
        for ( const auto &fixed : _fixedVariables )
        {
            if ( _uneliminableVariables.exists( fixed.first ) )
                continue;

            _preprocessed->_networkLevelReasoner->eliminateVariable( fixed.first, fixed.second );
        }
    }

    // Compute the new variable indices, after the elimination of fixed variables
    int offset = 0;
    unsigned numEliminated = 0;
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( ( _fixedVariables.exists( i ) || _mergedVariables.exists( i ) ) &&
             !_uneliminableVariables.exists( i ) )
        {
            ++numEliminated;
            ++offset;
        }
        else
            _oldIndexToNewIndex[i] = i - offset;
    }

    // Next, eliminate the fixed variables from the equations
    List<Equation> &equations( _preprocessed->getEquations() );
    List<Equation>::iterator equation = equations.begin();

    while ( equation != equations.end() )
    {
        // Each equation is of the form sum(addends) = scalar. So, any fixed variable
        // needs to be subtracted from the scalar. Merged variables should have already
        // been removed, so we don't care about them
        List<Equation::Addend>::iterator addend = equation->_addends.begin();
        while ( addend != equation->_addends.end() )
        {
            ASSERT( !_mergedVariables.exists( addend->_variable ) );

            if ( _fixedVariables.exists( addend->_variable ) )
            {
                // Addend has to go...
                double constant = _fixedVariables.at( addend->_variable ) * addend->_coefficient;
                equation->_scalar -= constant;
                addend = equation->_addends.erase( addend );
            }
            else
            {
                // Adjust the addend's variable index
                addend->_variable = _oldIndexToNewIndex.at( addend->_variable );
                ++addend;
            }
        }

        // If all the addends have been removed, we remove the entire equation.
        // Overwise, we are done here.
        if ( equation->_addends.empty() )
        {
            if ( _statistics )
                _statistics->incUnsignedAttribute( Statistics::PP_NUM_EQUATIONS_REMOVED );

            // No addends left, scalar should be 0
            if ( !FloatUtils::isZero( equation->_scalar ) )
                throw InfeasibleQueryException();
            else
                equation = equations.erase( equation );
        }
        else
            ++equation;
    }

    // Let the piecewise-linear constraints know of any eliminated variables, and remove
    // the constraints themselves if they become obsolete.
    List<PiecewiseLinearConstraint *> &constraints(
        _preprocessed->getPiecewiseLinearConstraints() );
    List<PiecewiseLinearConstraint *>::iterator constraint = constraints.begin();
    while ( constraint != constraints.end() )
    {
        List<unsigned> participatingVariables = ( *constraint )->getParticipatingVariables();
        for ( unsigned variable : participatingVariables )
        {
            if ( _uneliminableVariables.exists( variable ) )
                continue;

            if ( ( *constraint )->supportVariableElimination() &&
                 _fixedVariables.exists( variable ) )
                ( *constraint )->eliminateVariable( variable, _fixedVariables.at( variable ) );
        }

        if ( ( *constraint )->constraintObsolete() )
        {
            if ( _statistics )
                _statistics->incUnsignedAttribute( Statistics::PP_NUM_CONSTRAINTS_REMOVED );

            if ( _preprocessed->_networkLevelReasoner )
                _preprocessed->_networkLevelReasoner->removeConstraintFromTopologicalOrder(
                    *constraint );
            delete *constraint;
            *constraint = NULL;
            constraint = constraints.erase( constraint );
        }
        else
            ++constraint;
    }

    // Let the remaining piecewise-lienar constraints know of any changes in indices.
    for ( const auto &constraint : constraints )
    {
        List<unsigned> participatingVariables = constraint->getParticipatingVariables();
        for ( unsigned variable : participatingVariables )
        {
            if ( _oldIndexToNewIndex.at( variable ) != variable )
                constraint->updateVariableIndex( variable, _oldIndexToNewIndex.at( variable ) );
        }
    }

    // Let the nonlinear constraints know of any eliminated variables, and remove
    // the constraints themselves if they become obsolete.
    List<NonlinearConstraint *> &nlConstraints( _preprocessed->getNonlinearConstraints() );
    List<NonlinearConstraint *>::iterator nlConstraint = nlConstraints.begin();
    while ( nlConstraint != nlConstraints.end() )
    {
        List<unsigned> participatingVariables = ( *nlConstraint )->getParticipatingVariables();
        for ( unsigned variable : participatingVariables )
        {
            if ( _uneliminableVariables.exists( variable ) )
                continue;

            if ( ( *nlConstraint )->supportVariableElimination() &&
                 _fixedVariables.exists( variable ) )
            {
                ( *nlConstraint )->eliminateVariable( variable, _fixedVariables.at( variable ) );
            }
        }

        if ( ( *nlConstraint )->constraintObsolete() )
        {
            if ( _statistics )
                _statistics->incUnsignedAttribute( Statistics::PP_NUM_CONSTRAINTS_REMOVED );

            delete *nlConstraint;
            *nlConstraint = NULL;
            nlConstraint = nlConstraints.erase( nlConstraint );
        }
        else
            ++nlConstraint;
    }

    // Let the remaining nonlinear constraints know of any changes in indices.
    for ( const auto &nlConstraint : nlConstraints )
    {
        List<unsigned> participatingVariables = nlConstraint->getParticipatingVariables();
        for ( unsigned variable : participatingVariables )
        {
            if ( _oldIndexToNewIndex.at( variable ) != variable )
                nlConstraint->updateVariableIndex( variable, _oldIndexToNewIndex.at( variable ) );
        }
    }

    // Let the NLR know of changes in indices and merged variables
    if ( _preprocessed->_networkLevelReasoner )
        _preprocessed->_networkLevelReasoner->updateVariableIndices( _oldIndexToNewIndex,
                                                                     _mergedVariables );

    // Update the lower/upper bound maps
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( ( _fixedVariables.exists( i ) || _mergedVariables.exists( i ) ) &&
             !_uneliminableVariables.exists( i ) )
            continue;

        ASSERT( _oldIndexToNewIndex.at( i ) <= i );

        setLowerBound( _oldIndexToNewIndex.at( i ), getLowerBound( i ) );
        setUpperBound( _oldIndexToNewIndex.at( i ), getUpperBound( i ) );
    }

    // Adjust variable indices in the debugging solution
    Map<unsigned, double> copy = _preprocessed->_debuggingSolution;
    _preprocessed->_debuggingSolution.clear();
    for ( const auto &debugPair : copy )
    {
        unsigned variable = debugPair.first;
        double value = debugPair.second;

        // Go through any merges
        while ( variableIsMerged( variable ) )
            variable = getMergedIndex( variable );

        // Grab new index
        variable = getNewIndex( variable );

        _preprocessed->_debuggingSolution[variable] = value;
    }

    // Adjust the number of variables in the query
    _preprocessed->setNumberOfVariables( _preprocessed->getNumberOfVariables() - numEliminated );

    // Adjust the input/output mappings in the query
    _preprocessed->adjustInputOutputMapping( _oldIndexToNewIndex, _mergedVariables );
}

bool Preprocessor::variableIsFixed( unsigned index ) const
{
    return _fixedVariables.exists( index );
}

double Preprocessor::getFixedValue( unsigned index ) const
{
    return _fixedVariables.at( index );
}

bool Preprocessor::variableIsMerged( unsigned index ) const
{
    return _mergedVariables.exists( index );
}

unsigned Preprocessor::getMergedIndex( unsigned index ) const
{
    return _mergedVariables.at( index );
}

bool Preprocessor::variableIsUnusedAndSymbolicallyFixed( unsigned index ) const
{
    return _unusedSymbolicallyFixedVariables.exists( index );
}

unsigned Preprocessor::getNewIndex( unsigned oldIndex ) const
{
    if ( _oldIndexToNewIndex.exists( oldIndex ) )
        return _oldIndexToNewIndex.at( oldIndex );

    return oldIndex;
}

void Preprocessor::setStatistics( Statistics *statistics )
{
    _statistics = statistics;
}

void Preprocessor::setSolutionValuesOfEliminatedNeurons( InputQuery &inputQuery )
{
    Map<unsigned, double> assignment;
    for ( unsigned i = 0; i < inputQuery.getNumberOfVariables(); ++i )
    {
        if ( !_unusedSymbolicallyFixedVariables.exists( i ) )
            assignment[i] = inputQuery.getSolutionValue( i );
    }


    Map<unsigned, LinearExpression> unusedSymbolicallyFixedVariables =
        _unusedSymbolicallyFixedVariables;
    bool progressMade = true;
    List<unsigned> assignedVariables;
    while ( progressMade )
    {
        assignedVariables.clear();
        for ( auto &variableToExpression : unusedSymbolicallyFixedVariables )
        {
            unsigned var = variableToExpression.first;
            ASSERT( !assignment.exists( var ) );

            LinearExpression &exp = variableToExpression.second;
            double value = exp.evaluate( assignment );
            if ( !FloatUtils::isNan( value ) )
            {
                progressMade = true;
                assignment[var] = value;
                assignedVariables.append( var );
            }
        }
        for ( const auto &var : assignedVariables )
            unusedSymbolicallyFixedVariables.erase( var );
        progressMade = !assignedVariables.empty();
    }

    if ( !unusedSymbolicallyFixedVariables.empty() )
    {
        throw MarabouError( MarabouError::UNABLE_TO_RECONSTRUCT_SOLUTION_FOR_ELIMINATED_NEURONS );
    }

    for ( unsigned i = 0; i < inputQuery.getNumberOfVariables(); ++i )
    {
        inputQuery.setSolutionValue( i, assignment[i] );
    }
}

void Preprocessor::setMissingBoundsToInfinity()
{
    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        if ( !_preprocessed->getLowerBounds().exists( i ) )
            _preprocessed->setLowerBound( i, FloatUtils::negativeInfinity() );
        if ( !_preprocessed->getUpperBounds().exists( i ) )
            _preprocessed->setUpperBound( i, FloatUtils::infinity() );
    }
}

void Preprocessor::dumpAllBounds( const String &message )
{
    printf( "\nPP: Dumping all bounds (%s)\n", message.ascii() );

    for ( unsigned i = 0; i < _preprocessed->getNumberOfVariables(); ++i )
    {
        printf( "\tx%u: [%5.2lf, %5.2lf]\n", i, getLowerBound( i ), getUpperBound( i ) );
    }

    printf( "\n" );
}
