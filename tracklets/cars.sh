BAGPATH=/home/antor/didi-ext/didi-data/release3/Data-points/testing
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford01.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford02.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford03.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford04.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford05.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford06.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/ford07.bag
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 -b $BAGPATH/mustang01.bag
