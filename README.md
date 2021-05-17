# Body-pix demo
Testing body-pix and tensorflow with Rust.

## About
This is just a demo project to learn how to use Tensorflow with a frozen model.

## Usage
You have to download and convert the body-pix model, and put it in the
 `assets/models` folder. Update the name of the model and the stride
 accordingly.

### Getting the model
[simple_bodypix_python](https://github.com/ajaichemmanam/simple_bodypix_python)
 has a script (`get-model.sh`) that is a good tool for downloading models.

Run `pip install tfjs_graph_converter` to install the tool that you can use to
 convert body-pix to a frozen model.

To convert, run: `tfjs_graph_converter ./path/to/model/ output_model.pb`
