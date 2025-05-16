/**
 * Webpack Configuration for Super AI Frontend
 *
 * This configuration handles bundling, minification, and optimization
 * of JavaScript, CSS, and static assets for production.
 */

const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const CopyPlugin = require('copy-webpack-plugin');

// Define common paths
const PATHS = {
    src: path.resolve(__dirname, 'js'),
    dist: path.resolve(__dirname, 'dist'),
    css: path.resolve(__dirname, 'css'),
    assets: path.resolve(__dirname, 'assets')
};

module.exports = (env, argv) => {
    const isProduction = argv.mode === 'production';

    return {
        // Entry points for different bundles
        entry: {
            main: path.join(PATHS.src, 'main.js'),
            dashboard: path.join(PATHS.src, 'dashboard.js'),
            predictions: path.join(PATHS.src, 'predictions.js')
        },

        // Output configuration
        output: {
            path: PATHS.dist,
            filename: isProduction
                ? 'js/[name].[contenthash].js'
                : 'js/[name].js',
            publicPath: '/static/dist/'
        },

        // Development options
        devtool: isProduction ? 'source-map' : 'eval-source-map',

        // Optimization settings
        optimization: {
            minimize: isProduction,
            minimizer: [
                // Minify JavaScript
                new TerserPlugin({
                    terserOptions: {
                        compress: {
                            drop_console: isProduction,
                        },
                        format: {
                            comments: false,
                        },
                    },
                    extractComments: false,
                }),
                // Minify CSS
                new CssMinimizerPlugin(),
            ],
            // Split code into chunks
            splitChunks: {
                chunks: 'all',
                name: false,
                cacheGroups: {
                    vendors: {
                        test: /[\\/]node_modules[\\/]/,
                        name: 'vendors',
                        chunks: 'all',
                    },
                    commons: {
                        name: 'commons',
                        chunks: 'all',
                        minChunks: 2,
                    },
                },
            },
        },

        // Module rules
        module: {
            rules: [
                // JavaScript/ES6 processing
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                            plugins: [
                                '@babel/plugin-proposal-class-properties',
                                '@babel/plugin-proposal-object-rest-spread'
                            ]
                        }
                    }
                },
                // CSS processing
                {
                    test: /\.css$/,
                    use: [
                        MiniCssExtractPlugin.loader,
                        'css-loader',
                        {
                            loader: 'postcss-loader',
                            options: {
                                postcssOptions: {
                                    plugins: [
                                        'autoprefixer',
                                        'cssnano'
                                    ]
                                }
                            }
                        }
                    ]
                },
                // Images
                {
                    test: /\.(png|svg|jpg|jpeg|gif)$/i,
                    type: 'asset',
                    parser: {
                        dataUrlCondition: {
                            maxSize: 8 * 1024 // 8kb
                        }
                    },
                    generator: {
                        filename: 'images/[name].[hash][ext][query]'
                    }
                },
                // Fonts
                {
                    test: /\.(woff|woff2|eot|ttf|otf)$/i,
                    type: 'asset/resource',
                    generator: {
                        filename: 'fonts/[name].[hash][ext][query]'
                    }
                }
            ]
        },

        // Plugins
        plugins: [
            // Clean output directory before building
            new CleanWebpackPlugin(),

            // Extract CSS into separate files
            new MiniCssExtractPlugin({
                filename: isProduction
                    ? 'css/[name].[contenthash].css'
                    : 'css/[name].css'
            }),

            // Copy static assets
            new CopyPlugin({
                patterns: [
                    {
                        from: path.join(PATHS.assets, 'favicon.ico'),
                        to: PATHS.dist
                    },
                    {
                        from: path.join(PATHS.assets, 'images'),
                        to: path.join(PATHS.dist, 'images'),
                        noErrorOnMissing: true
                    }
                ],
            }),
        ],

        // Performance hints
        performance: {
            hints: isProduction ? 'warning' : false,
            maxAssetSize: 512000, // 500 KiB
            maxEntrypointSize: 512000, // 500 KiB
        },

        // Development server
        devServer: {
            contentBase: PATHS.dist,
            compress: true,
            port: 9000,
            hot: true,
            proxy: {
                '/api': 'http://localhost:8888'
            }
        }
    };
};
