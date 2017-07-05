# CHANGE TO THE PATH THAT CONTAINS THE BAGS 
BAGPATH=/home/antor/didi-ext/didi-data/release3/Data-points/testing
#BAGPATH=/home/kaliev/catkin_ws/src/didi_pipeline/tracklets/testing
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford01.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford02.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford03.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford04.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford05.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford06.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ford07.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/mustang01.bag
