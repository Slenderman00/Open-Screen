#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h> 
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

class OpenScreenCpp {
    public: 
        OpenScreenCpp() : running(false) {}
        const char* model_path;
        //model input size:
        const int input_size = 224;
        //background path
        const char* background_path;

		PyObject* start(PyObject* self, PyObject* args) {
			const char* model_path = nullptr;
			const char* background_path = nullptr;
			int input_size;
		
			if (!PyArg_ParseTuple(args, "sis", &model_path, &input_size, &background_path)) {
				return NULL;
			}
		
			if (running) {
				Py_RETURN_NONE;
			}
		
			running = true;
			camera_thread = std::thread(&OpenScreenCpp::camera_thread_fn, this);
			inference_thread = std::thread(&OpenScreenCpp::inference_thread_fn, this);
			camera_output_thread = std::thread(&OpenScreenCpp::camera_output_thread_fn, this);
		
			Py_RETURN_NONE;
		}

		PyObject* stop(PyObject* self, PyObject* args) {
			running = false;
		
			if (camera_thread.joinable()) {
				camera_thread.join();
			}
			if (inference_thread.joinable()) {
				inference_thread.join();
			}
			if (camera_output_thread.joinable()) {
				camera_output_thread.join();
			}
		
			Py_RETURN_NONE; //return None to indicate success
		}

    private: 
        std::atomic<bool> running;
        std::thread camera_thread;
        std::thread inference_thread;
        std::thread camera_output_thread;

        //buffer for the camera thread
        cv::Mat buffer;
        
        //current mask from the inference thread
        cv::Mat mask;

        //the camera thread reads images from the camera into the buffer
        void camera_thread_fn() {
            cv::VideoCapture cap(0);
            if(!cap.isOpened()) {
                return;
            }
            while(running) {
                cap >> buffer;
            }
        }

        //the inference thread runs the model on the images in the buffer
        void inference_thread_fn() {
            torch::jit::script::Module module;
            try {
                module = torch::jit::load(model_path);
            } catch(const c10::Error& e) {
                return;
            }
            while(running) {
                cv::Mat image;
                buffer.copyTo(image);
                cv::resize(image, image, cv::Size(input_size, input_size));
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                torch::Tensor tensor_image = torch::from_blob(image.data, {input_size, input_size, 3}, torch::kByte);
                tensor_image = tensor_image.permute({2, 0, 1});
                tensor_image = tensor_image.toType(torch::kFloat);
                tensor_image = tensor_image.div(255);
                tensor_image = tensor_image.unsqueeze(0);
                at::Tensor output = module.forward({tensor_image}).toTensor();
                output = output.squeeze(0);
                output = output.argmax(0);
                output = output.toType(torch::kByte);
                output = output.mul(255);
                output = output.toType(torch::kU8);
                cv::Mat mask(output.size(0), output.size(1), CV_8UC1, output.data_ptr());
                cv::resize(mask, mask, buffer.size());
                mask.copyTo(mask);
            }
        }

        //the camera output thread displays the camera image with the mask overlayed and replaced with the background
        void camera_output_thread_fn() {
            cv::Mat background = cv::imread(background_path);
            while(running) {
                cv::Mat image;
                buffer.copyTo(image);
                cv::Mat output;
                cv::addWeighted(image, 0.5, mask, 0.5, 0, output);
                //cv::imshow("output", output);
                //cv::waitKey(1);
            }
        }

};

static PyObject* start(PyObject* self, PyObject* args) {
	return OpenScreenCpp().start(self, args);
}

static PyObject* stop(PyObject* self, PyObject* args) {
	return OpenScreenCpp().stop(self, args);
}

static PyMethodDef OpenScreenCppMethods[] = {
	{"start", start, METH_VARARGS, "Start the OpenScreenCpp"},
	{"stop", stop, METH_VARARGS, "Stop the OpenScreenCpp"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef OpenScreenCppModule = {
	PyModuleDef_HEAD_INIT,
	"OpenScreenCpp",
	NULL,
	-1,
	OpenScreenCppMethods
};

PyMODINIT_FUNC PyInit_OpenScreenCpp(void) {
	return PyModule_Create(&OpenScreenCppModule);
}