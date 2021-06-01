# Body-pix demo
Testing body-pix and tensorflow with Rust.

## About
This is just a demo project to learn how to use Tensorflow with a frozen model.

## Usage
Build the application and start it. Use the control panel to open and process
 images.

## Dependencies
The build-script depends on python3 and the package `tfjs_graph_converter`. It
 should be available through a simple installation with `pip`:  
`pip install tfjs_graph_converter`

The build script will then download and convert the body-pix models

## Thanks
A big shout-out to Ajai Chemmanam (ajaichemmanam on Github), and his
 simple_bodypix_python project. I owe my understanding of how to massage the
 input and output to improve prediction to this project. (Which I have been
 tinkering with a lot.) Check it out here:
[simple_bodypix_python](https://github.com/ajaichemmanam/simple_bodypix_python)
