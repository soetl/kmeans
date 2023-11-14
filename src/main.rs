use std::collections::HashSet;

use polars::prelude::*;
use rand::Rng;

const CLUSTER_CENTER_1: u64 = 7;
const CLUSTER_CENTER_2: u64 = 18;
const RESULT_DIRECTORY: &str = "./result";

fn main() {
    let df = LazyCsvReader::new("kmeans.csv")
        .has_header(true)
        .finish()
        .unwrap()
        .select([
            col("n").cast(DataType::UInt64),
            col("*").exclude(&["n"]).cast(DataType::Float64),
        ]);

    let csv_options = CsvWriterOptions {
        has_header: true,
        batch_size: 10000,
        maintain_order: false,
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

    let kmeans = KMeans::new(df, 2 as u8, Some(vec![CLUSTER_CENTER_1, CLUSTER_CENTER_2]), csv_options.clone());

    for (i, lf) in kmeans.eval().iter().enumerate() {
        let _ = lf.clone().sink_csv(
            format!("{RESULT_DIRECTORY}/res_{}.csv", i).into(),
            csv_options.clone(),
        );
    }
}

struct KMeans {
    df: LazyFrame,
    clusters: Vec<LazyFrame>,
    centers: Vec<Vec<f64>>,
    csv_options: CsvWriterOptions,
}

impl KMeans {
    pub fn new(df: LazyFrame, n_clusters: impl Into<usize>, center_ids: Option<Vec<u64>>, csv_options: CsvWriterOptions) -> Self {
        let n_clusters = n_clusters.into();
        let centers = Self::init_centers(df.clone(), n_clusters, center_ids);

        KMeans {
            df,
            centers,
            clusters: Vec::new(),
            csv_options,
        }
    }

    fn eval(mut self) -> Vec<LazyFrame> {
        let mut clusters_last;
        self.clusters = vec![self.df.clone()];

        let mut step = 1;

        loop {
            println!("Step {}", step);
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

            let _ = df_clusters.clone().sink_csv(
                format!("{RESULT_DIRECTORY}/{}_dist.csv", step).into(),
                self.csv_options.clone(),
            );

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
                .select(&[col("*").exclude(&["x", "y", "z", "n"])])
                .collect()
                .unwrap()
                .iter()
                .map(|s| s.f64().unwrap().into_no_null_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let mut cluster_tags = Vec::new();

            for i in 0..df_num.len() {
                let mut min_dist = f64::MAX;
                let mut min_dist_idx = 0;

                for j in 0..clusters_dist.len() {
                    let dist = clusters_dist[j].get(i).unwrap();
                    let dist = *dist;

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

            let _ = df.clone().sink_csv(
                format!("{RESULT_DIRECTORY}/{}_clusters.csv", step).into(),
                self.csv_options.clone(),
            );

            self.clusters = df
                .collect()
                .unwrap()
                .partition_by(&["cluster"], false)
                .unwrap()
                .into_iter()
                .map(|x| x.lazy())
                .collect();

            self.eval_centers();

            if self.clusters[0]
                .clone()
                .collect()
                .unwrap()
                .eq(&clusters_last[0].clone().collect().unwrap())
            {
                return self.clusters;
            }

            step += 1;
        }
    }

    fn eval_centers(&mut self) {
        let mut centers = Vec::<Vec<f64>>::new();

        for lf in self.clusters.clone() {
            centers.push(
                lf.collect()
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
            .select(&[col("*").exclude(&["n"])])
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
