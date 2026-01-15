# Human-Demonstration-Sidetracks

# How to collect humn feedback
1. Build using ```catkin_make```
2. open two terminals and do the regualar source
3. launch the simulation using 
```bash
roslaunch sim-env launch_sim.launch
```
4. launch the replayer by using
```bash
roslaunch replayer gui.launch bag:={the bag number you want to replay}
```

# Note:
1. If you want to quit a replay, simply ctrl+C in the replayer terminal. The arm will atomatically return to home.
2. all annotations will be stored in 
```
src/replayer/scripts/annotations
```
as JSON files.
3. The annotations_by_expert are the annotations done by experts and was analyzed in the paper.