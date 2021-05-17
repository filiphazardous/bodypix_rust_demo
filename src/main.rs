mod bodypix;
mod image_utils;

use bodypix::*;
use image_utils::*;

use nannou::image::io::Reader as ImageReader;
use nannou::math::cgmath::{Matrix4, SquareMatrix};
use nannou::prelude::*;
use nannou::wgpu::Texture;

use tensorflow::{Session, SessionOptions};

fn main() {
    println!("Tensorflow version: {}", tensorflow::version().unwrap());
    nannou::app(model).update(update).simple_window(view).run();
}

struct Model {
    image_texture: Option<Texture>,
    mask_texture: Option<Texture>,
    silhouette_texture: Option<Texture>,
    cutout_texture: Option<Texture>,
}

fn model<'a>(app: &App) -> Model {
    app.set_loop_mode(LoopMode::Wait);

    let (mut graph, stride) = load_bodypix_model(app.assets_path().unwrap());
    let mut session = Session::new(&SessionOptions::new(), &graph).unwrap();

    let test_image = ImageReader::open(
        app.assets_path()
            .unwrap()
            .join("images")
            .join("test-image.jpg"),
    )
    .unwrap()
    .decode()
    .unwrap();
    let mask = image_to_mask(&mut graph, &mut session, &test_image, stride);
    let mask_image = mask_to_image(&mask);
    let silhouette_image = create_silhouette(&mask, &test_image);
    let cutout_image = create_cutout(&mask, &test_image);

    let image_texture = Some(Texture::from_image(app, &test_image));
    let mask_texture = Some(Texture::from_image(app, &mask_image));
    let silhouette_texture = Some(Texture::from_image(app, &silhouette_image));
    let cutout_texture = Some(Texture::from_image(app, &cutout_image));

    Model {
        image_texture,
        mask_texture,
        silhouette_texture,
        cutout_texture,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn mirror_mat() -> Matrix4<f32> {
    let mut matrix: Matrix4<f32> = Matrix4::identity();
    matrix[0][0] = -1.;
    matrix
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(DIMGREY);
    let draw = app.draw();
    let mirror: Matrix4<f32> = mirror_mat();

    match &model.mask_texture {
        Some(m) => {
            draw.scale(0.75)
                .x_y(335., 260.)
                .transform(mirror)
                .texture(&m);
            ()
        }
        None => (),
    }

    match &model.image_texture {
        Some(i) => {
            draw.scale(0.75)
                .x_y(-335., 260.)
                .transform(mirror)
                .texture(&i);
            ()
        }
        None => (),
    }

    match &model.silhouette_texture {
        Some(m) => {
            draw.scale(0.75)
                .x_y(335., -260.)
                .transform(mirror)
                .texture(&m);
            ()
        }
        None => (),
    }

    match &model.cutout_texture {
        Some(i) => {
            draw.scale(0.75)
                .x_y(-335., -260.)
                .transform(mirror)
                .texture(&i);
            ()
        }
        None => (),
    }

    draw.to_frame(app, &frame).unwrap();
}
