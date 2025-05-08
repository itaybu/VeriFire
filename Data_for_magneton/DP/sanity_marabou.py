from maraboupy import Marabou
import numpy as np

marabou_net = Marabou.read_tf("dp-net", modelType="savedModel_v2", outputNames=['StatefulPartitionedCall/StatefulPartitionedCall/sequential/dense_1/BiasAdd'])
# for i in range(25*25):
#     marabou_net.setLowerBound(x=i, v=1)
#     marabou_net.setUpperBound(x=i, v=1)
print(marabou_net.evaluateWithMarabou(np.ones((1,25,25,1))))

# marabou_net.saveQuery("ipq_with_sigmoid")
# marabou_net.solve()

pass