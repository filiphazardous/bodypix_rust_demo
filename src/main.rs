mod bodypix;

use bodypix::*;

use nannou::image::io::Reader as ImageReader;
use nannou::math::cgmath::{Matrix4, SquareMatrix};
use nannou::prelude::*;
use nannou::ui::color::WHITE;
use nannou::ui::prelude::*;
use nannou::wgpu::Texture;

use tensorflow::{Session, SessionOptions};

fn main() {
    println!("Tensorflow version: {}", tensorflow::version().unwrap());
    nannou::app(model).update(update).simple_window(view).run();
}

struct Model {
    ui: Ui,
    ids: Ids,

    image_texture: Option<Texture>,
    mask_texture: Option<Texture>,
}

widget_ids! {
    struct Ids {
        button,
    }
}

fn model<'a>(app: &App) -> Model {
    let mut ui = app.new_ui().build().unwrap();
    app.set_loop_mode(LoopMode::RefreshSync);
    let ids = Ids::new(ui.widget_id_generator());

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
    let mask_image = image_to_mask(&mut graph, &mut session, &test_image, stride);

    let image_texture = Some(Texture::from_image(app, &test_image));
    let mask_texture = Some(Texture::from_image(app, &mask_image));

    Model {
        ui,
        ids,

        image_texture,
        mask_texture,
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    let Model {
        ref mut ui, ids, ..
    } = model;

    let ui = &mut ui.set_widgets();

    if widget::Button::new()
        .top_left()
        .w_h(200.0, 30.0)
        .border(0.0)
        .color(WHITE)
        .label("Click")
        .label_font_size(15)
        .set(ids.button, ui)
        .was_clicked()
    {
        println!("Click!");
    }
}

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

    draw.to_frame(app, &frame).unwrap();
    model.ui.draw_to_frame(app, &frame).unwrap();
}
