use std::collections::HashSet;

use plotters::prelude::*;
use polars::prelude::*;
use rand::Rng;

const RESULT_DIRECTORY: &str = "./result";

fn main() {
    if std::path::Path::new(&RESULT_DIRECTORY).exists() {
        std::fs::remove_dir_all(RESULT_DIRECTORY).expect("Failed to remove result directory");
    }

    std::fs::create_dir_all(RESULT_DIRECTORY).expect("Failed to create result directory");

    let df = LazyCsvReader::new("kmeans.csv")
        .has_header(true)
        .finish()
        .unwrap()
        .select([
            col("n").cast(DataType::UInt64),
            col("*").exclude(["n"]).cast(DataType::Float64),
        ]);

    let csv_options = CsvWriterOptions {
        has_header: true,
        batch_size: 10000,
        maintain_order: true,
        serialize_options: SerializeOptions {
            date_format: None,
            time_format: None,
            datetime_format: None,
            float_precision: Some(5),
            separator: b","[0],
            quote_char: b"~"[0],
            null: "None".to_owned(),
            line_terminator: "\n".to_owned(),
            quote_style: QuoteStyle::Necessary,
        },
    };

    let mut dann_indexes = Vec::new();
    for i in 2..15_u8 {
        let kmeans = KMeans::new(df.clone(), i, None, false, csv_options.clone());
        let clusters = kmeans.eval();
        if clusters.len() != i as usize {
            println!("{} num of clusters decreased to {}", i, clusters.len());
        } else {
            let dann_index = dann_index(clusters.clone());
            dann_indexes.push((i, dann_index));
        }
    }

    let mut max = (0_u8, f64::MIN);
    dann_indexes.iter().for_each(|(x, y)| if *y > max.1 { max = (*x, *y) });

    let kmeans = KMeans::new(df.clone(), max.0, None, true, csv_options.clone());

    for (i, lf) in kmeans.eval().iter().enumerate() {
        let _ = lf.clone().sink_csv(
            format!("{RESULT_DIRECTORY}/res_{}_cluster.csv", i).into(),
            csv_options.clone(),
        );
    }
    
    let chart_path = format!("{RESULT_DIRECTORY}/dann_index.png");
    let root = BitMapBackend::new(&chart_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Dann Index over Clusters", ("sans-serif", 40))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(1..16, 0.0..1.0)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(30)
        .max_light_lines(4)
        .y_desc("Score")
        .draw()
        .unwrap();

    let _ = chart.draw_series(LineSeries::new(
        dann_indexes
            .iter()
            .map(|(x, y)| (*x as i32, *y))
            .collect::<Vec<_>>(),
        &RED,
    ));

    let _ = chart.draw_series(
        dann_indexes
            .iter()
            .map(|(x, y)| Circle::new((*x as i32, *y), 3, BLUE.filled())),
    );
}

#[derive(Clone)]
struct KMeans {
    df: LazyFrame,
    clusters: Vec<LazyFrame>,
    centers: Vec<Vec<f64>>,
    io: bool,
    csv_options: CsvWriterOptions,
}

impl KMeans {
    pub fn new(
        df: LazyFrame,
        n_clusters: impl Into<usize>,
        center_ids: Option<Vec<u64>>,
        io: bool,
        csv_options: CsvWriterOptions,
    ) -> Self {
        let n_clusters = n_clusters.into();
        let centers = Self::init_centers(df.clone(), n_clusters, center_ids);

        KMeans {
            df,
            centers,
            clusters: Vec::new(),
            io,
            csv_options,
        }
    }

    fn eval(mut self) -> Vec<LazyFrame> {
        let mut clusters_last;
        self.clusters = vec![self.df.clone()];

        let mut step = 1;

        loop {
            clusters_last = self.clusters.clone();

            let mut exprs = Vec::new();
            for i in 0..self.centers.len() {
                exprs.push(
                    ((col("x") - lit(self.centers[i][0])).pow(2)
                        + (col("y") - lit(self.centers[i][1])).pow(2)
                        + (col("z") - lit(self.centers[i][2])).pow(2))
                    .sqrt()
                    .alias(format!("cluster{}dist", i).as_str()),
                );
            }

            let df_clusters = self.df.clone().with_columns(exprs);

            if self.io {
                let _ = df_clusters.clone().sink_csv(
                    format!("{RESULT_DIRECTORY}/{}__dist.csv", step).into(),
                    self.csv_options.clone(),
                );
            }

            let df_num = df_clusters
                .clone()
                .select(&[col("n")])
                .collect()
                .unwrap()
                .iter()
                .map(|s| s.u64().unwrap().into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let df_num = df_num[0].clone();

            let clusters_dist = df_clusters
                .clone()
                .select(&[col("*").exclude(["x", "y", "z", "n"])])
                .collect()
                .unwrap()
                .iter()
                .map(|s| s.f64().unwrap().into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let mut cluster_tags = Vec::new();

            for i in 0..df_num.len() {
                let mut min_dist = f64::MAX;
                let mut min_dist_idx = 0;

                for (j, dist) in clusters_dist.iter().enumerate() {
                    let dist = dist[i];

                    if dist < min_dist {
                        min_dist = dist;
                        min_dist_idx = j;
                    }
                }

                cluster_tags.push(min_dist_idx as u64);
            }

            let s1 = Series::new("n", df_num);
            let s2 = Series::new("cluster", cluster_tags);

            let df_clusters = DataFrame::new(vec![s1, s2]).unwrap().lazy();

            let df = self.df.clone().left_join(df_clusters, col("n"), col("n"));

            self.clusters = df
                .collect()
                .unwrap()
                .partition_by(["cluster"], false)
                .unwrap()
                .into_iter()
                .map(|x| x.lazy())
                .collect();

            if self.io {
                for (i, lf) in self.clusters.iter().enumerate() {
                    let _ = lf.clone().sink_csv(
                        format!("{RESULT_DIRECTORY}/{}_{}_cluster.csv", step, i).into(),
                        self.csv_options.clone(),
                    );
                }
            }

            self.eval_centers();

            step += 1;

            if self.centers.len() <= 1 {
                return self.clusters;
            }

            if self.clusters.len() != clusters_last.len() {
                continue;
            }

            let mut count = 0;
            for i in 0..self.clusters.len() {
                for j in &clusters_last {
                    if self.clusters[i]
                        .clone()
                        .collect()
                        .unwrap()
                        .eq(&j.clone().collect().unwrap())
                    {
                        count += 1;
                    }
                }
            }

            if count == self.clusters.len() {
                return self.clusters;
            }

            if self.clusters[0]
                .clone()
                .collect()
                .unwrap()
                .eq(&clusters_last[0].clone().collect().unwrap())
            {
                return self.clusters;
            }
        }
    }

    fn eval_centers(&mut self) {
        let mut centers = Vec::<Vec<f64>>::new();

        for lf in self.clusters.clone() {
            centers.push(
                lf.select([col("*").exclude(["n"])])
                    .collect()
                    .unwrap()
                    .iter()
                    .map(|s| s.sum::<f64>().unwrap() / s.len() as f64)
                    .collect(),
            );
        }

        self.centers = centers;
    }

    fn init_centers(
        df: LazyFrame,
        n_clusters: usize,
        center_ids: Option<Vec<u64>>,
    ) -> Vec<Vec<f64>> {
        let height = { df.clone().collect().unwrap().height() };

        let center_ids = match center_ids {
            Some(center_ids) => center_ids.into_iter().collect::<Series>(),
            None => {
                let mut rng = rand::thread_rng();
                let mut center_ids = HashSet::new();

                while center_ids.len() < n_clusters {
                    center_ids.insert(rng.gen_range(0..height as u64));
                }

                center_ids.into_iter().collect()
            }
        };

        let centers_df = df
            .clone()
            .filter(col("n").is_in(lit(center_ids)))
            .select(&[col("*").exclude(["n"])])
            .collect()
            .unwrap();

        let mut centers = Vec::<Vec<f64>>::new();
        for i in 0..centers_df.height() {
            centers.push(Vec::new());

            for j in centers_df.get(i).unwrap() {
                centers[i].push(j.try_extract::<f64>().unwrap());
            }
        }

        centers
    }
}

fn dann_index(lf: Vec<LazyFrame>) -> f64 {
    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for i in 0..lf.len() {
        let lf1 = lf[i].clone();

        let s = lf1
            .clone()
            .select([col("*").exclude(["n"])])
            .collect()
            .unwrap()
            .iter()
            .map(|s| s.f64().cloned().unwrap())
            .collect::<Vec<_>>();

        for j in 0..s[0].len() {
            for k in 0..s[0].len() {
                if j == k {
                    continue;
                }

                let mut sum = 0.0;
                for l in s.clone() {
                    sum += (l.get(j).unwrap() - l.get(k).unwrap()).powi(2);
                }

                let d = sum.sqrt();

                if d > max {
                    max = d;
                }
            }
        }

        for (j, lf2) in lf.iter().enumerate() {
            if i == j {
                continue;
            }

            let s2 = lf2
                .clone()
                .select([col("*").exclude(["n"])])
                .collect()
                .unwrap()
                .iter()
                .map(|s| s.f64().cloned().unwrap())
                .collect::<Vec<_>>();

            for k in 0..s[0].len() {
                for l in 0..s2[0].len() {
                    let mut sum = 0.0;
                    for m in 0..s.len() {
                        sum += (s[m].get(k).unwrap() - s2[m].get(l).unwrap()).powi(2);
                    }

                    let d = sum.sqrt();

                    if d < min {
                        min = d;
                    }
                }
            }
        }
    }

    min / max
}
