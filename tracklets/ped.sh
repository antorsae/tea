BAGPATH=/home/antor/didi-ext/didi-data/release3/Data-points/testing
rosrun didi_pipeline ros_node.py -sm $1 -lm $2 -di -rfp -lpt 3 $3 -b $BAGPATH/ped_test.bag
