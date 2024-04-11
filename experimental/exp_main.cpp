#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

#include "GLViewer.hpp"
#include "yolo.hpp"

#include <sl/Camera.hpp>
#include <NvInfer.h>

using namespace nvinfer1;
//
#define NMS_THRESH 0.4
#define CONF_THRESH 0.75

#define LAUNCH_HEIGHT 1.55

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

cv::Rect get_rect(BBox box) {
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

std::vector<sl::uint2> cvt(const BBox &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: \n 1. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine\n 2. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine images:1x3x512x512\n 3. ./yolo_onnx_zed yolov8s.engine <SVO path>" << std::endl;
        return 0;
    }
    
    // Check Optim engine first
    if (std::string(argv[1]) == "-s" && (argc >= 4)) {
        std::string onnx_path = std::string(argv[2]);
        std::string engine_path = std::string(argv[3]);
        OptimDim dyn_dim_profile;

        if (argc == 5) {
            std::string optim_profile = std::string(argv[4]);
            bool error = dyn_dim_profile.setFromString(optim_profile);
            if (error) {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return EXIT_FAILURE;
            }
        }

        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl:RESOLUTION::HD720;
    init_parameters.camera_fps = sl::CAMERA_FPS::60;
    init_parameters.grab_compute_capping_fps = sl::GRAB_COMPUTE_CAPPING_FPS::10;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = sl::COORDINATE_UNITS::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    std::cout << "Parameters Set" << std::endl;

    if (argc > 1) {
        std::string zed_opt = argv[2];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Camera Open" << returned_state << "\nExit program.";
        return EXIT_FAILURE;
    }
    // Positional Tracking
    sl::PositionalTrackingParameters tracking_parameters;
    tracking_parameters.set_floor_as_orign = sl::SET_FLOOR_AS_ORIGIN::true;
    tracking_parameters.reference_frame = sl::REFERENCE_FRAME::WORLD;
    zed.enablePositionalTracking(tracking_parameters);
    
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = false; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);
    // Failure of Object Detection Initialization
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "enableObjectDetection " << returned_state << "\nExit program.";
        zed.close();
        return EXIT_FAILURE;
    }
    
    // Configure Camera and Display
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;
    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, true);
    // ---------

    // Creating the inference engine class
    std::string engine_name = "";
    Yolo detector;
    if (argc > 0)
        engine_name = argv[1];
    else {
        std::cout << "Error: missing engine name as argument" << std::endl;
        return EXIT_FAILURE;
    }
    if (detector.init(engine_name)) {
        std::cerr << "Detector init failed!" << std::endl;
        return EXIT_FAILURE;
    }

    auto display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl, point_cloud;
    cv::Mat left_cv;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();
    
    //-----------------------------------------------------------------------------------
    //
    // Beginning of launch
    //

    // Set launch status and get initial position
    int launch_status = -1;
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
        POSITIONAL_TRACKING_STATE state = zed.getPostion(cam_w_pose, REFERENCE_FRAME::WORLD);
    }

    // Begin camera feed and flight
    while (viewer.isAvailable()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

            // Initiating Launch
            if(launch_status == -1) {
                std::cout << "Drone launching to a height of " << LAUNCH_HEIGHT << "m" << std::endl;
                // droneLaunch(DRONE_HEIGHT);
                // Show camera feed until launch height reached
                while (cam_w_pose.getTranslation().tz < (LAUNCH_HEIGHT-0.15)) {
                    // Get image for inference
                    zed.retrieveImage(left_sl, sl::VIEW::LEFT);
                    // Get image for display
                    left_cv = slMat2cvMat(left_sl);
                    // Display image
                    cv::imshow("Launch", left_cv);
                    cv::waitKey(10);
                    // GL Viewer
                    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
                    zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
                    viewer.updateData(point_cloud, cam_w_pose.pose_data);
                }
                // Change launch status for successful launch
                launch_status = 0 
            }
            
            // Begin Object Detection and Searching

            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Running inference
            auto detections = detector.run(left_sl, display_resolution.height, display_resolution.width, CONF_THRESH);

            // Get image for display
            left_cv = slMat2cvMat(left_sl);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : detections) {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int) it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space                
                objects_in.push_back(tmp);
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);


            // Displaying 'raw' objects
            for (size_t j = 0; j < detections.size(); j++) {
                cv::Rect r = get_rect(detections[j].box);
                cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(left_cv, std::to_string((int) detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imshow("Objects", left_cv);
            cv::waitKey(10);

            // Print for debugging detection behavior
            print("Detections: ", detections.size(), ".\n")

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            // GL Viewer
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            viewer.updateData(point_cloud, objects.object_list, cam_w_pose.pose_data);

            // 1 Object Detected
            if (detections.size() == 1 && launch_status == 1) {
                // Change launch status to reflect detected object
                launch_status = 2;
                std::cout << "Landing pad detected: " << objects.object_list.front().position() << std::endl;

                sl::float3 pad_position = objects.object_list.front().position();
                float cam_x = cam_w_pose.getTranslation().tx;
                float cam_y = cam_w_pose.getTranslation().ty;
                std::cout << "Camera Position: " << cam_w_pose.getTranslation() << std::endl;
                std::cout << "Camera Orientation: " << cam_w_pose.getOrientation() << std::endl;
                //fly_right(pad_position[0] - cam_x);    //Flies right necessary x distance (will fly left if it is negative)
                //fly_forward(pad_position[1] - cam_y);  //Flies forward necessary y distance
                std::cout << "Fly Right: " << (pad_position[0] - cam_x) << std::endl;
                std::cout << "Fly Forward: " << (pad_position[1] - cam_y) << std::endl;

                while (cam_w_pose.getTranslation().ty < (pad_position[1] - 0.1)) {
                    // Get image for inference
                    zed.retrieveImage(left_sl, sl::VIEW::LEFT);
                    // Get image for display
                    left_cv = slMat2cvMat(left_sl);
                    // Display image
                    cv::imshow("Launch", left_cv);
                    cv::waitKey(10);
                    // GL Viewer
                    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
                    zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
                    viewer.updateData(point_cloud, cam_w_pose.pose_data);
                }

                // Drone is above landing pad - initiate landing
                std::cout << "Landing " << std::endl;
                //triggerLanding();
                while (cam_w_pose.getTranslation().tz > 0.2) {
                    // Get image for inference
                    zed.retrieveImage(left_sl, sl::VIEW::LEFT);
                    // Get image for display
                    left_cv = slMat2cvMat(left_sl);
                    // Display image
                    cv::imshow("Launch", left_cv);
                    cv::waitKey(10);
                    // GL Viewer
                    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
                    zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
                    viewer.updateData(point_cloud, cam_w_pose.pose_data);
                }

                //Set launch status to indicate a sucessful landing
                launch_status = 3;
            }
            
            if (launch_status == 3) {
                std::cout << "Mission Complete" << std::endl;
                viewer.exit();
                return 0;
            }
        }
    }

    viewer.exit();
    return 0;
}
