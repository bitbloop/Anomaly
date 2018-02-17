# Semi-Supervised Anomaly Detection in N-Dimensional Data.

## Summary

Using semi-supervised learning and Python, given a data set with mostly valid examples, we learn the distribution of the points using Gaussian and Multivariate Gaussian models. Then we can flag points as anomalous or not using the trained models.

The supervised part of the algorithm is specifying the threshold value which is used to flag the points.

## Figure

The displayed graphs below represent a Gaussian Model (top) and a Multivariate Gaussian Model (bottom). Both models were trained using the same input data.

![Anomaly](http://radosjovanovic.com/projects/git/anomaly.png)

Graph Legend:  
x and y axis - Data points in 2D space  

Blue Points - Input Data which is used to build a model  
Red Points - Points in 2D space flagged as anomalous  
Yellow Points - Points in 2D space flagged as valid  

## Authors

* **Rados Jovanovic** - *Initial work* - [bitbloop](https://github.com/bitbloop)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to everyone contributing to science!


