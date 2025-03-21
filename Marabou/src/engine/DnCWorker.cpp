/*********************                                                        */
/*! \file DnCWorker.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Haoze Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "DnCWorker.h"

#include "Debug.h"
#include "EngineState.h"
#include "IEngine.h"
#include "LargestIntervalDivider.h"
#include "MStringf.h"
#include "MarabouError.h"
#include "PiecewiseLinearCaseSplit.h"
#include "PolarityBasedDivider.h"
#include "SnCDivideStrategy.h"
#include "SubQuery.h"
#include "TableauStateStorageLevel.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

DnCWorker::DnCWorker( WorkerQueue *workload,
                      std::shared_ptr<IEngine> engine,
                      std::atomic_int &numUnsolvedSubQueries,
                      std::atomic_bool &shouldQuitSolving,
                      unsigned threadId,
                      unsigned onlineDivides,
                      float timeoutFactor,
                      SnCDivideStrategy divideStrategy,
                      unsigned verbosity,
                      bool parallelDeepSoI,
                      std::atomic<double> &percentageCompleted )
    : _workload( workload )
    , _engine( engine )
    , _numUnsolvedSubQueries( &numUnsolvedSubQueries )
    , _shouldQuitSolving( &shouldQuitSolving )
    , _threadId( threadId )
    , _onlineDivides( onlineDivides )
    , _timeoutFactor( timeoutFactor )
    , _verbosity( verbosity )
    , _parallelDeepSoI( parallelDeepSoI )
    , _percentageCompleted( &percentageCompleted )
{
    setQueryDivider( divideStrategy );

    // Obtain the current state of the engine
    if ( !_parallelDeepSoI )
    {
        _initialState = std::make_shared<EngineState>();
        _engine->storeState( *_initialState, TableauStateStorageLevel::STORE_ENTIRE_TABLEAU_STATE );
    }
}

void DnCWorker::setQueryDivider( SnCDivideStrategy divideStrategy )
{
    if ( divideStrategy == SnCDivideStrategy::Polarity )
        _queryDivider = std::unique_ptr<QueryDivider>( new PolarityBasedDivider( _engine ) );
    else
    {
        const List<unsigned> &inputVariables = _engine->getInputVariables();
        _queryDivider =
            std::unique_ptr<LargestIntervalDivider>( new LargestIntervalDivider( inputVariables ) );
    }
}

void DnCWorker::popOneSubQueryAndSolve( bool restoreTreeStates )
{
    SubQuery *subQuery = NULL;
    // Boost queue stores the next element into the passed-in pointer
    // and returns true if the pop is successful (aka, the queue is not empty
    // in most cases)
    if ( _workload->pop( subQuery ) )
    {
        String queryId = subQuery->_queryId;
        unsigned depth = subQuery->_depth;
        auto split = std::move( subQuery->_split );
        std::unique_ptr<SmtState> smtState = nullptr;
        if ( restoreTreeStates && subQuery->_smtState )
            smtState = std::move( subQuery->_smtState );
        unsigned timeoutInSeconds = subQuery->_timeoutInSeconds;

        // Reset the engine state
        _engine->restoreState( *_initialState );
        _engine->reset();

        // TODO: each worker is going to keep a map from *CaseSplit to an
        // object of class DnCStatistics, which contains some basic
        // statistics. The maps are owned by the DnCManager.

        // Apply the split and solve
        _engine->applySnCSplit( *split, queryId );

        bool fullSolveNeeded = true; // denotes whether we need to solve the subquery
        if ( restoreTreeStates && smtState )
            fullSolveNeeded = _engine->restoreSmtState( *smtState );
        IEngine::ExitCode result = IEngine::NOT_DONE;
        if ( fullSolveNeeded )
        {
            _engine->solve( timeoutInSeconds );
            result = _engine->getExitCode();
        }
        else
        {
            // UNSAT is proven when replaying stack-entries
            result = IEngine::UNSAT;
        }

        if ( _verbosity > 0 )
            printProgress( queryId, result );
        // Switch on the result
        if ( result == IEngine::UNSAT )
        {
            // If UNSAT, continue to solve
            *_numUnsolvedSubQueries -= 1;
            if ( _numUnsolvedSubQueries->load() == 0 || _parallelDeepSoI )
                *_shouldQuitSolving = true;
            delete subQuery;

            double subQueryPercentage = 1;
            auto bounds = split->getBoundTightenings();

            if ( _engine->getInputQuery() )
            {
                for ( unsigned var : _engine->getInputVariables() )
                {
                    double origLb = _engine->getInputQuery()->getLowerBound( var );
                    double origUb = _engine->getInputQuery()->getUpperBound( var );

                    double lb = 0;
                    double ub = 0;

                    for ( const Tightening &tightening : bounds )
                    {
                        if ( tightening._variable == var )
                        {
                            if ( tightening._type == Tightening::LB )
                                lb = tightening._value;
                            else
                                ub = tightening._value;
                        }
                    }

                    subQueryPercentage *= ( ub - lb ) / ( origUb - origLb );
                }

                double prevPercentage = _percentageCompleted->load();
                while ( !_percentageCompleted->compare_exchange_weak(
                    prevPercentage, prevPercentage + subQueryPercentage ) )
                {
                }

                printf( "Search Tree Percentage Completed: %f\n", _percentageCompleted->load() );
            }
        }
        else if ( result == IEngine::TIMEOUT )
        {
            // If TIMEOUT, split the current input region and add the
            // new subQueries to the current queue
            SubQueries subQueries;
            unsigned newTimeout = ( depth >= GlobalConfiguration::DNC_DEPTH_THRESHOLD - 1
                                        ? 0
                                        : (unsigned)timeoutInSeconds * _timeoutFactor );
            unsigned numNewSubQueries = pow( 2, _onlineDivides );
            std::vector<std::unique_ptr<SmtState>> newSmtStates;
            if ( restoreTreeStates )
            {
                // create |numNewSubQueries| copies of the current SmtState
                for ( unsigned i = 0; i < numNewSubQueries; ++i )
                {
                    newSmtStates.push_back( std::unique_ptr<SmtState>( new SmtState() ) );
                    _engine->storeSmtState( *( newSmtStates[i] ) );
                }
            }

            _queryDivider->createSubQueries(
                numNewSubQueries, queryId, depth, *split, newTimeout, subQueries );

            unsigned i = 0;
            for ( auto &newSubQuery : subQueries )
            {
                // Store the SmtCore state
                if ( restoreTreeStates )
                {
                    newSubQuery->_smtState = std::move( newSmtStates[i++] );
                }

                if ( !_workload->push( std::move( newSubQuery ) ) )
                {
                    throw MarabouError( MarabouError::UNSUCCESSFUL_QUEUE_PUSH );
                }

                *_numUnsolvedSubQueries += 1;
            }
            *_numUnsolvedSubQueries -= 1;
            delete subQuery;
        }
        else if ( result == IEngine::QUIT_REQUESTED )
        {
            // If engine was asked to quit, quit
            std::cout << "Quit requested by manager!" << std::endl;
            delete subQuery;
            ASSERT( _shouldQuitSolving->load() );
        }
        else
        {
            // We must set the quit flag to true  if the result is not UNSAT or
            // TIMEOUT. This way, the DnCManager will kill all the DnCWorkers.

            *_shouldQuitSolving = true;
            if ( result == IEngine::SAT )
            {
                // case SAT
                *_numUnsolvedSubQueries -= 1;
                delete subQuery;
            }
            else if ( result == IEngine::ERROR )
            {
                // case ERROR
                std::cout << "Error!" << std::endl;
                delete subQuery;
            }
            else // result == IEngine::NOT_DONE
            {
                // case NOT_DONE
                ASSERT( false );
                std::cout << "Not done! This should not happen." << std::endl;
                delete subQuery;
            }
        }
    }
    else
    {
        // If the queue is empty but the pop fails, wait and retry
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }
}

void DnCWorker::printProgress( String queryId, IEngine::ExitCode result ) const
{
    printf( "Worker %d: Query %s %s, %d tasks remaining\n",
            _threadId,
            queryId.ascii(),
            exitCodeToString( result ).ascii(),
            _numUnsolvedSubQueries->load() );
}

String DnCWorker::exitCodeToString( IEngine::ExitCode result )
{
    switch ( result )
    {
    case IEngine::UNSAT:
        return "unsat";
    case IEngine::SAT:
        return "sat";
    case IEngine::ERROR:
        return "ERROR";
    case IEngine::TIMEOUT:
        return "TIMEOUT";
    case IEngine::QUIT_REQUESTED:
        return "QUIT_REQUESTED";
    default:
        ASSERT( false );
        return "UNKNOWN (this should never happen)";
    }
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
