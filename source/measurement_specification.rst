Measurement Specification using YAML Files
==========================================

While you can access the features QuPyt offers a number of ways, the recommended one is to use the qupyt CLI, and specify your measurement
using yaml files. 

When you start QuPyt by typing ``qupyt`` on you commandline, QuPyt will perform some setup operations if necessary.
Afterwards it will start to observe the directory (folder) ``~/.qupyt/waiting_room/`` to check for any new files
created or moved there. If it detects new yaml files, it loads and pareses them for instructions
on the next measurement to perform. How these yaml files are created is entirely up to the end user. We provide
some sample files that should be sufficient to configure your own measurement. 
