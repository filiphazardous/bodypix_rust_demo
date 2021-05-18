mod bodypix;
mod image_utils;

use bodypix::*;
use image_utils::*;

use nannou::image::io::Reader as ImageReader;
use nannou::prelude::*;
use nannou::ui::prelude::*;
use nannou::wgpu::Texture;

use dirs::picture_dir;
use nannou::ui::widget::file_navigator::Types::WithExtension;
use std::path::PathBuf;

fn main() {
    println!("Tensorflow version: {}", tensorflow::version().unwrap());
    nannou::app(model).update(update).simple_window(view).run();
}

widget_ids! {
    struct Ids {
        toggle_file_picker,
        file_picker,
        process_image,
    }
}

struct Model {
    ui: Ui,
    ids: Ids,

    show_file_picker: bool,
    image_file_path: PathBuf,

    body_pix: BodyPix,

    textures: Option<(Texture, Texture, Texture, Texture)>, // image, mask, silhouette, cutout
}

fn model<'a>(app: &App) -> Model {
    let mut ui = app.new_ui().build().unwrap();
    app.set_loop_mode(LoopMode::Wait);
    let ids = Ids::new(ui.widget_id_generator());

    let body_pix = BodyPix::from_model(
        app.assets_path()
            .unwrap()
            .join("models")
            .join("bodypix_resnet50_float_model-stride16.pb"),
    );

    Model {
        ui,
        ids,

        show_file_picker: false,
        image_file_path: PathBuf::new(),

        body_pix,

        textures: (None),
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let Model {
        ref mut ui,
        ids,
        show_file_picker,
        image_file_path,
        body_pix,
        ..
    } = model;

    let ui = &mut ui.set_widgets();

    let toggle_fp_label = if *show_file_picker {
        "Cancel"
    } else {
        "Pick image"
    };
    if widget::Button::new()
        .top_left()
        .w_h(150., 30.)
        .label(toggle_fp_label)
        .set(ids.toggle_file_picker, ui)
        .was_clicked()
    {
        *show_file_picker = !*show_file_picker;
    }

    if *show_file_picker {
        for selection in widget::FileNavigator::new(
            &*picture_dir().unwrap(),
            WithExtension(&["jpg", "jpeg", "png", "gif"]),
        )
        .w_h(1000., 700.)
        .set(ids.file_picker, ui)
        {
            use nannou::ui::widget::file_navigator::Event::ChangeSelection;
            match selection {
                ChangeSelection(vec_pb) => {
                    if vec_pb.len() > 0 {
                        let path = vec_pb.last().unwrap();
                        if path.is_file() {
                            let p_string = path.clone().into_os_string().into_string().unwrap();
                            println!("Change selection: {}", p_string);
                            *image_file_path = path.clone();
                            *show_file_picker = false;
                        }
                    }
                }
                _ => {}
            }
        }
    } else if image_file_path.is_file()
        && widget::Button::new()
            .label("Process image")
            .set(ids.process_image, ui)
            .was_clicked()
    {
        let test_image = ImageReader::open(image_file_path)
            .unwrap()
            .decode()
            .unwrap();
        let mask = body_pix.process_image(&test_image);
        let mask_image = mask_to_image(&mask);
        let silhouette_image = create_silhouette(&mask, &test_image);
        let cutout_image = create_cutout(&mask, &test_image);

        let image_texture = Texture::from_image(app, &test_image);
        let mask_texture = Texture::from_image(app, &mask_image);
        let silhouette_texture = Texture::from_image(app, &silhouette_image);
        let cutout_texture = Texture::from_image(app, &cutout_image);

        model.textures = Some((
            image_texture,
            mask_texture,
            silhouette_texture,
            cutout_texture,
        ));
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(DIMGREY);
    let draw = app.draw();

    if model.textures.is_some() {
        let (image_texture, mask_texture, silhouette_texture, cutout_texture) =
            model.textures.as_ref().unwrap();
        draw.x_y(310., 230.)
            .texture(&mask_texture)
            .w_h(400., 300.)
            .finish();

        draw.x_y(-90., 230.)
            .texture(&image_texture)
            .w_h(400., 300.)
            .finish();

        draw.x_y(310., -70.)
            .texture(&silhouette_texture)
            .w_h(400., 300.)
            .finish();

        draw.x_y(-90., -70.)
            .texture(&cutout_texture)
            .w_h(400., 300.)
            .finish();
    }

    draw.to_frame(app, &frame).unwrap();
    model.ui.draw_to_frame(app, &frame).unwrap();
}
