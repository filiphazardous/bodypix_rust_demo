mod bodypix;
mod image_utils;

use bodypix::*;
use image_utils::*;

use nannou::image::io::Reader as ImageReader;
use nannou::prelude::*;
use nannou::ui::prelude::*;
use nannou::wgpu::Texture;

use dirs::picture_dir;
use nannou::image::DynamicImage;
use nannou::ui::widget::file_navigator::Types::WithExtension;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    nannou::app(model).update(update).run();
}

widget_ids! {
    struct Ids {
        toggle_file_picker,
        file_picker,
        process_image,
        bodypix_model,
    }
}

struct Model {
    ui: Ui,
    main_window: WindowId,
    cp_window: WindowId,
    ids: Ids,

    show_file_picker: bool,
    image_file_path: PathBuf,

    body_pix: Option<BodyPix>,

    selected_image: Option<DynamicImage>,
    image_texture: Option<Texture>,
    textures: Option<(Texture, Texture, Texture)>, // mask, silhouette, cutout

    selected_model: Option<usize>,
}

fn model<'a>(app: &App) -> Model {
    app.set_loop_mode(LoopMode::Wait);

    let main_window = app
        .new_window()
        .title(app.exe_name().unwrap())
        .size(1280, 960)
        .view(view)
        .build()
        .unwrap();

    let cp_window = app
        .new_window()
        .title(app.exe_name().unwrap() + " controls")
        .size(640, 480)
        .view(cp_view)
        .event(cp_event)
        .build()
        .unwrap();

    let mut ui = app.new_ui().window(cp_window).build().unwrap();
    let ids = Ids::new(ui.widget_id_generator());

    Model {
        ui,
        ids,
        main_window,
        cp_window,

        show_file_picker: false,
        image_file_path: PathBuf::new(),

        body_pix: (None),

        selected_image: (None),
        image_texture: (None),
        textures: (None),

        selected_model: (None),
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let Model {
        ref mut ui,
        ids,
        show_file_picker,
        image_file_path,
        body_pix,

        selected_image,
        image_texture,

        selected_model,
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
        .w_h(640., 420.)
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

                            *selected_image = Some(
                                ImageReader::open(&image_file_path)
                                    .unwrap()
                                    .decode()
                                    .unwrap(),
                            );

                            *image_texture =
                                Some(Texture::from_image(app, selected_image.as_ref().unwrap()));

                            *show_file_picker = false;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let bpx_model_data = BodyPix::models();
    let model_drop_down = widget::DropDownList::new(bpx_model_data.0.as_ref(), *selected_model)
        .w_h(200.0, 30.0)
        .label("BodyPix model");

    for selected_idx in model_drop_down.set(ids.bodypix_model, ui) {
        *selected_model = Some(selected_idx);

        let model_file_name = bpx_model_data.1[selected_idx];
        let stride = bpx_model_data.2[selected_idx];

        println!("Selected model: {}, {}", model_file_name, stride);

        let model_path = app
            .assets_path()
            .unwrap()
            .join("models")
            .join(model_file_name);

        let model_type = bpx_model_data.3[selected_idx];
        let body_pix_init = BodyPix::from_model(model_path, stride, model_type);

        *body_pix = Some(body_pix_init);
    }

    if image_file_path.is_file()
        && selected_model.is_some()
        && widget::Button::new()
            .label("Process image")
            .set(ids.process_image, ui)
            .was_clicked()
    {
        let t = Instant::now();

        let mask = body_pix.as_ref().unwrap().process_image(
            selected_image.as_ref().unwrap(),
            InterpolationType::LinearMean,
        );

        let t_delta_1 = t.elapsed().as_micros() as f32 / 1000.;
        println!("Time to process: {}", t_delta_1);

        let mask_image = mask_to_image(&mask);
        let silhouette_image = create_silhouette(&mask, selected_image.as_ref().unwrap());
        let cutout_image = create_cutout(&mask, selected_image.as_ref().unwrap());

        let t_delta_2 = t.elapsed().as_micros() as f32 / 1000. - t_delta_1;
        println!("Time to create images: {}", t_delta_2);

        let mask_texture = Texture::from_image(app, &mask_image);
        let silhouette_texture = Texture::from_image(app, &silhouette_image);
        let cutout_texture = Texture::from_image(app, &cutout_image);

        let t_delta_total = t.elapsed().as_micros() as f32 / 1000.;
        let t_delta_3 = t_delta_total - t_delta_1 - t_delta_2;
        println!("Time to create textures: {}", t_delta_3);

        println!("Total time to process: {}", t_delta_total);

        model.textures = Some((mask_texture, silhouette_texture, cutout_texture));
    }
}

fn cp_event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn cp_view(app: &App, model: &Model, frame: Frame) {
    frame.clear(DIMGREY);
    model.ui.draw_to_frame(app, &frame).unwrap();
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(DIMGREY);
    let draw = app.draw();

    if model.image_texture.is_some() {
        let image_texture = model.image_texture.as_ref().unwrap();

        draw.x_y(-320., 240.)
            .texture(&image_texture)
            .w_h(400., 300.)
            .finish();
    }

    if model.textures.is_some() {
        let (mask_texture, silhouette_texture, cutout_texture) = model.textures.as_ref().unwrap();

        draw.x_y(320., 240.)
            .texture(&mask_texture)
            .w_h(400., 300.)
            .finish();

        draw.x_y(320., -240.)
            .texture(&silhouette_texture)
            .w_h(400., 300.)
            .finish();

        draw.x_y(-320., -240.)
            .texture(&cutout_texture)
            .w_h(400., 300.)
            .finish();
    }

    draw.to_frame(app, &frame).unwrap();
}
