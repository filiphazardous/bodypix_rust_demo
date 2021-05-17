# Body-pix demo
Testing body-pix and tensorflow with Rust.

## About
This is just a demo project to learn how to use Tensorflow with a frozen model.

## Usage
You have to download and convert the body-pix model, and put it in the
 `assets/models` folder. Update the name of the model and the stride
 accordingly.

Put an image called `test-image.jpg` in `assets/images` (or with some other
 name - in which case you have to update the name).

### Getting the model
[simple_bodypix_python](https://github.com/ajaichemmanam/simple_bodypix_python)
 has a script (`get-model.sh`) that is a good tool for downloading models.

Run `pip install tfjs_graph_converter` to install the tool that you can use to
 convert body-pix to a frozen model.

To convert, run: `tfjs_graph_converter ./path/to/model/ output_model.pb`

## Thanks
A big shout-out to Ajai Chemmanam (ajaichemmanam on Github), and his
 simple_bodypix_python project. I owe my understanding of how to massage the
 input and output to improve prediction to this project. (Which I have been
 tinkering with a lot.)
