# Decision Trees (and Forests)

This is a decision forest library written in C++. It consists of a header file `dtree.hpp` and a cpp file `dtree.cpp`. The library is tested using the file called `test.cpp`.  

The test program takes a training data file and the test data file via the command line, as well as a third argument which decides if it uses a decision tree/forest or if the attributes are picked for optimizing tree depth or at random.
The statistics are then spat out in the out.txt file.

There are some popular datasets included. I have tested them and the model's accuracy is in the 80's for all of them except the yeast dataset.

# Running the programs
Language: C++ (C++17 standard)

To compile: Make sure to use a C++17 compatible compiler. Command:
```shell
g++ dtree.cpp hw3.cpp -o dtree -Wall --std=c++17
```
