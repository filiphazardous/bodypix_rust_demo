use std::fs::File;
use std::io::Read;
use std::ops::Index;
use std::path::PathBuf;

use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

use nannou::image;
use nannou::image::{DynamicImage, GenericImageView, Pixel};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum ModelType {
    MobileNet,
    ResNet,
}

pub struct BodyPix {
    graph: Graph,
    session: Session,
    stride: u32,
    model_type: ModelType,
    // input_operations
    // output_operations
}

pub struct Segments {
    pub orig_width: usize,
    pub orig_height: usize,
    width: usize,
    height: usize,
    stride: u32,
    values: Vec<f32>,
}

impl Segments {
    fn sigmoid(val: f32) -> f32 {
        let neg_val = -val;
        let denominator = 1. + neg_val.exp();
        1. / denominator
    }

    pub fn from_tensor(stride: u32, orig_width: usize, orig_height: usize, t: &Tensor<f32>) -> Segments {
        let height = t.shape().index(1).unwrap() as usize;
        let width = t.shape().index(2).unwrap() as usize;

        let mut values = vec![0f32; width * height];

        // Is there a faster way to copy all the elements, and treat them? Or should we just move the sigmoid function to a shader and make one single copy?
        for x in 0..width {
            for y in 0..height {
                let seg_val = t.get(&[0, y as u64, x as u64, 0]);
                values[x + y * width] = Segments::sigmoid(seg_val);
            }
        }
        let values = values;

        Segments {
            orig_width,
            orig_height,
            width,
            height,
            stride,
            values,
        }
    }

    pub fn no_interpolation(self: &Segments, x: usize, y: usize) -> f32 {
        let step_x = 1 + x / self.stride as usize;
        let step_y = 1 + y / self.stride as usize;
        self.values[step_x + step_y * self.width]
    }

    pub fn linear_mean(self: &Segments, x: usize, y: usize) -> f32 {
        let step_x = x / self.stride as usize;
        let step_y = y / self.stride as usize;

        let x1: usize = step_x;
        let x2: usize = if x1 >= self.width - 1 {
            x1
        } else {
            1 + step_x
        };

        let y1 = step_y;
        let y2 = if y1 >= self.height - 1 {
            y1
        } else {
            1 + step_y
        };

        let x_remainder = x as u32 % self.stride;
        let y_remainder = y as u32 % self.stride;

        let part_x1 = (self.stride - x_remainder) as f32 / self.stride as f32;
        let part_x2 = 1. - part_x1;
        let part_y1 = (self.stride - y_remainder) as f32 / self.stride as f32;
        let part_y2 = 1. - part_y1;

        let mean_x1 = self.values[x1 + y1 * self.width] * part_y1 + self.values[x1 + y2 * self.width] * part_y2;
        let mean_x2 = self.values[x2 + y1 * self.width] * part_y1 + self.values[x2 + y2 * self.width] * part_y2;

        let mean = mean_x1 * part_x1 + mean_x2 * part_x2;

        mean
    }

}

impl BodyPix {
    pub fn models() -> (
        Vec<&'static str>,
        Vec<&'static str>,
        Vec<u32>,
        Vec<ModelType>,
    ) {
        (
            vec![
                "MobileNet 0.50 stride 8",
                "MobileNet 0.50 stride 16",
                "MobileNet 0.75 stride 8",
                "MobileNet 0.75 stride 16",
                "MobileNet 1.00 stride 8",
                "MobileNet 1.00 stride 16",
                "ResNet 0.50 stride 16",
                "ResNet 0.50 stride 32",
            ],
            vec![
                "bodypix_mobilenet_float_050-stride8.pb",
                "bodypix_mobilenet_float_050-stride8.pb",
                "bodypix_mobilenet_float_075-stride8.pb",
                "bodypix_mobilenet_float_075-stride8.pb",
                "bodypix_mobilenet_float_100-stride8.pb",
                "bodypix_mobilenet_float_100-stride8.pb",
                "bodypix_resnet50_float-stride16.pb",
                "bodypix_resnet50_float-stride32.pb",
            ],
            vec![8, 16, 8, 16, 8, 16, 16, 32],
            vec![
                ModelType::MobileNet,
                ModelType::MobileNet,
                ModelType::MobileNet,
                ModelType::MobileNet,
                ModelType::MobileNet,
                ModelType::MobileNet,
                ModelType::ResNet,
                ModelType::ResNet,
            ],
        )
    }

    pub fn from_model(model_path: PathBuf, stride: u32, model_type: ModelType) -> BodyPix {
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
            model_type,
        }
    }

    pub fn process_image(&self, image: &DynamicImage) -> Segments {
        let orig_width = image.width();
        let orig_height = image.height();

        // Nudge target width/height one step above stride (improves quality!)
        let target_width = orig_width + 1;
        let target_height = orig_height + 1;

        let max_x = orig_width - 1;
        let max_y = orig_height - 1;

        // Setup tensor args
        let vec_size = target_width * target_height * 3;
        let mut flattened: Vec<f32> = Vec::with_capacity(vec_size as usize); // TODO: Maybe have a reusable buffer, instead of re-allocating every frame?

        match self.model_type {
            ModelType::MobileNet => {
                for y in 0..target_height {
                    for x in 0..target_width {
                        let pixel = image.get_pixel(std::cmp::min(x, max_x), std::cmp::min(y, max_y)).to_rgb();

                        let red = pixel.0[0] as f32 / 127.5 - 1.;
                        let green = pixel.0[1] as f32 / 127.5 - 1.;
                        let blue = pixel.0[2] as f32 / 127.5 - 1.;

                        flattened.push(red);
                        flattened.push(green);
                        flattened.push(blue);
                    }
                }
            }
            ModelType::ResNet => {
                for y in 0..target_height {
                    for x in 0..target_width {
                        let pixel = image.get_pixel(std::cmp::min(x, max_x), std::cmp::min(y, max_y)).to_rgb();

                        let red = pixel.0[0] as f32 - 123.15;
                        let green = pixel.0[1] as f32 - 115.9;
                        let blue = pixel.0[2] as f32 - 103.06;

                        flattened.push(red);
                        flattened.push(green);
                        flattened.push(blue);
                    }
                }
            }
        }

        assert_eq!(flattened.len(), vec_size as usize);

        let input = Tensor::new(&[1, target_height as u64, target_width as u64, 3])
            .with_values(&flattened)
            .unwrap();

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

        let segments = Segments::from_tensor(self.stride, orig_width as usize, orig_height as usize, &segments_res);

        segments
    }
}
