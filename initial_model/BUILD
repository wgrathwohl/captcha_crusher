# Description:
# Example TensorFlow models for captcha

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "captcha_input",
    srcs = ["captcha_input.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "captcha_input_test",
    srcs = ["captcha_input_test.py"],
    deps = [
        ":captcha_input",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "captcha",
    srcs = ["captcha.py"],
    deps = [
        ":captcha_input",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "captcha_eval",
    srcs = [
        "captcha_eval.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":captcha",
    ],
)

py_binary(
    name = "captcha_train",
    srcs = [
        "captcha_train.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":captcha",
    ],
)

py_binary(
    name = "captcha_multi_gpu_train",
    srcs = [
        "captcha_multi_gpu_train.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":captcha",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
