# Python Robotics

A object-based toolbox for robot dynamic simulation, analysis, control and planning. 

## Installation ##

### Dependencies ####
* numpy
* scipy
* matplotlib

### Recommended environment ###
Anaconda distribution + spyder IDE available here: https://www.anaconda.com/products/individual

Note: If graphical animations are not working, try changing the graphics backend. In spyder this option is found in the menu at python/Preferences/IPython console/Backend. Inline does not allow animations, it is best to use Automatic (for Windows and Ubuntu) or OS X (for Mac).

### Clone repo and add to python path ###

A simple option for development is simply to clone the repo:
```bash
git clone https://github.com/SherbyRobotics/pyro.git
```
then add the pyro folder to the pythonpath variable of your environment. In spyder this option is found in the menu at python/PYTHONPATH manager.

## Library Architecture ##

### Dynamic objects ###

At the core of pyro is a mother-class representing generic non-linear dynamic systems, with the following nomemclature:

<img width="929" alt="Screen Shot 2021-05-02 at 15 57 47" src="https://user-images.githubusercontent.com/16725496/116826021-fd9b7a80-ab5f-11eb-8e50-d7361094cbee.png">

Other more specific mother-class are 
-Linear System
-Mechanical System
-Manipulator Robot

<img width="763" alt="Screen Shot 2021-05-02 at 16 13 51" src="https://user-images.githubusercontent.com/16725496/116826418-dc3b8e00-ab61-11eb-9372-09ae08f0b15a.png">


### Controller objects ###

Controller objects can be used to closed the loop with an operation generating a closed-loop dynamic system:

closed-loop system = controller + open-loop system

For "memoryless" controller, this operation is

<img width="760" alt="Screen Shot 2021-05-02 at 16 17 34" src="https://user-images.githubusercontent.com/16725496/116826519-59ff9980-ab62-11eb-8256-6a9f4a3f4f0f.png">

Available control algorithms: PID, LQR, Computed-Torque, End-point Impedance, Value-Iteration, Sliding-mode controller, etc.

### Trajectory planner objects ###

Cooming soon..


## How to use ##

See exemples scripts in pyro/examples.

Coming soon..

## Gallery of exemples ##

### Phase-plane Analysis ###

<img width="887" alt="Screen Shot 2021-05-02 at 16 41 44" src="https://user-images.githubusercontent.com/16725496/116827083-59b4cd80-ab65-11eb-9be9-b02586a89d7f.png">

### Optimal controller computation with Value-Iteration ###

<img width="1131" alt="Screen Shot 2021-05-02 at 16 42 34" src="https://user-images.githubusercontent.com/16725496/116827109-6c2f0700-ab65-11eb-95e7-5290e4b50b32.png">


### Car parallel parking solved with RRT, Value-Iteration, etc.. ###

<img width="889" alt="Screen Shot 2021-05-02 at 16 38 59" src="https://user-images.githubusercontent.com/16725496/116827009-ef9c2880-ab64-11eb-8d81-12a2d6453eac.png">

<img width="1167" alt="Screen Shot 2021-05-02 at 16 39 42" src="https://user-images.githubusercontent.com/16725496/116827025-0773ac80-ab65-11eb-8a1b-9b50086d86cb.png">

### Redondant Manipulator Controller ###

<img width="1154" alt="Screen Shot 2021-05-02 at 16 26 47" src="https://user-images.githubusercontent.com/16725496/116826685-46a0fe00-ab63-11eb-9f9f-f269aa0e63b5.png">

### Pendulums Swing-up solved with Computed-Torque, RRT, Value-Iteration, etc.. ###

<img width="1163" alt="Screen Shot 2021-05-02 at 16 34 04" src="https://user-images.githubusercontent.com/16725496/116826866-3c333400-ab64-11eb-9c99-e87742d273f7.png">

<img width="1148" alt="Screen Shot 2021-05-02 at 16 32 13" src="https://user-images.githubusercontent.com/16725496/116826821-ff673d00-ab63-11eb-969c-72c0d0711076.png">








