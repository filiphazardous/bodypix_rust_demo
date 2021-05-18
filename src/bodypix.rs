use std::fs::File;
use std::io::Read;
use std::ops::Index;
use std::path::PathBuf;

use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

use nannou::image;
use nannou::image::{DynamicImage, GenericImageView, Pixel};

pub struct BodyPix {
    graph: Graph,
    session: Session,
    stride: u32,
    // input_operations
    // output_operations
    // enum for resnet or mobilenet
}

impl BodyPix {
    pub fn from_model(model_path: PathBuf) -> BodyPix {
        let stride: u32 = 16;
        let mut model_data: Vec<u8> = Vec::new();
        let mut file = File::open(model_path).unwrap();
        file.read_to_end(&mut model_data).unwrap();

        let mut graph = Graph::new();
        graph
            .import_graph_def_with_results(&model_data, &ImportGraphDefOptions::new())
            .unwrap();

        let session = Session::new(&SessionOptions::new(), &graph).unwrap();

        BodyPix {
            stride,
            graph,
            session,
        }
    }

    pub fn blocky_sig(x: usize, y: usize, stride: u32, sig_map: &Vec<Vec<f32>>) -> f32 {
        let step_x = 1 + x / stride as usize;
        let step_y = 1 + y / stride as usize;
        sig_map[step_x][step_y]
    }

    pub fn mean_sig(x: usize, y: usize, stride: u32, sig_map: &Vec<Vec<f32>>) -> f32 {
        let step_x = x / stride as usize;
        let step_y = y / stride as usize;

        let x1: usize = step_x;
        let x2: usize = 1 + step_x;

        let y1 = step_y;
        let y2 = 1 + step_y;

        let x_remainder = x as u32 % stride;
        let y_remainder = y as u32 % stride;

        let part_x1 = (stride - x_remainder) as f32 / stride as f32;
        let part_x2 = 1. - part_x1;
        let part_y1 = (stride - y_remainder) as f32 / stride as f32;
        let part_y2 = 1. - part_y1;

        let mean_x1 = sig_map[x1][y1] * part_y1 + sig_map[x1][y2] * part_y2;
        let mean_x2 = sig_map[x2][y1] * part_y1 + sig_map[x2][y2] * part_y2;

        let mean = mean_x1 * part_x1 + mean_x2 * part_x2;

        mean
    }

    pub fn process_image(&self, image: &DynamicImage) -> Vec<Vec<f32>> {
        // TODO: Add mean-func as arg above (either blocky or mean)
        fn sigmoid(val: f32) -> f32 {
            let neg_val = -val;
            let denominator = 1. + neg_val.exp();
            1. / denominator
        }

        let img_width = image.width();
        let img_height = image.height();

        let target_width = img_width + 1;
        let target_height = img_height + 1;

        // TODO: This resize is superfluous - it could just as well be done in the conversion loop
        let input_image = image.resize_exact(target_width, target_height, image::imageops::Nearest);

        let target_width = input_image.width();
        let target_height = input_image.height();

        println!(
            "{}x{}: {}x{}",
            img_height, img_width, target_height, target_width
        );

        // Setup tensor args
        let vec_size = target_width * target_height * 3;
        let mut flattened: Vec<f32> = Vec::with_capacity(vec_size as usize);

        for y in 0..target_height {
            for x in 0..target_width {
                // TODO: If x == img_width, just get the pixel for (x-1) - and we've merged resize and flatten
                let pixel = input_image.get_pixel(x as u32, y as u32).to_rgb();

                // TODO: Check which kind of model we have here
                // Resnet preprocessing
                let red = pixel.0[0] as f32 - 123.15;
                let green = pixel.0[1] as f32 - 115.9;
                let blue = pixel.0[2] as f32 - 103.06;

                /*
                // Mobilenet preprocessing
                let red = pixel.0[0] as f32 / 127.5 - 1.;
                let green = pixel.0[1] as f32 / 127.5 - 1.;
                let blue = pixel.0[2] as f32 / 127.5 - 1.;
                */
                flattened.push(red);
                flattened.push(green);
                flattened.push(blue);
            }
        }
        assert_eq!(flattened.len(), vec_size as usize);

        let input = Tensor::new(&[1, target_height as u64, target_width as u64, 3])
            .with_values(&flattened)
            .unwrap();

        println!("Input Image Shape in hwc: {}", input.shape());
        let width_resolution = (input.shape()[1].unwrap() - 1) as u32 / self.stride + 1;
        let height_resolution = (input.shape()[0].unwrap() - 1) as u32 / self.stride + 1;
        println!("Resolution: {} {}", width_resolution, height_resolution);

        let mut args: SessionRunArgs = SessionRunArgs::new();
        args.add_feed(
            &self.graph.operation_by_name_required("sub_2").unwrap(),
            0,
            &input,
        );

        let segments_token = args.request_fetch(
            &self
                .graph
                .operation_by_name_required("float_segments")
                .unwrap(),
            0,
        );

        self.session.run(&mut args).unwrap();

        let segments_res: Tensor<f32> = args.fetch(segments_token).unwrap();

        let segment_height = segments_res.shape().index(1).unwrap() as usize;
        let segment_width = segments_res.shape().index(2).unwrap() as usize;

        let mut sig_map: Vec<Vec<f32>> = vec![vec![0f32; segment_height]; segment_width];
        for x in 0..segment_width {
            for y in 0..segment_height {
                let seg_val = segments_res.get(&[0, y as u64, x as u64, 0]);
                sig_map[x][y] = sigmoid(seg_val);
            }
        }
        let sig_map = sig_map;

        let mut mask: Vec<Vec<f32>> = vec![vec![0.; img_height as usize]; img_width as usize];
        for x in 0..img_width {
            for y in 0..img_height {
                let sig_val = BodyPix::mean_sig(x as usize, y as usize, self.stride, &sig_map);
                mask[x as usize][y as usize] = sig_val;
            }
        }

        mask
    }
}
