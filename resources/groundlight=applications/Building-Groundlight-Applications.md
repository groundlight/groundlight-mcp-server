# How to Build a Groundlight Application

Groundlight applications are Agentic processes that monitor video streams in the background for you, to monitor for specified events or conditions.  

## Architecture

The components to a full system are:
- A camera
- The Groundlight application code
- The Groundlight edge inference server (optional)
- The Groundlight cloud service

Cameras are typically networked cameras, but could be local USB cameras.  Applications use the `framegrab` library to abstract the camera interface, so that the same code can be used for local cameras, streaming cameras, and cameras behind NVRs.  The only difference is in the `cameras.yaml` configuration file, which defines the camera sources.  When using a managed Groundlight Hub device, the hub will provide the `cameras.yaml` file for the application, using its built-in camera-discovery UI.  But for development purposes, authoring a short stand-alone `cameras.yaml` file is appropriate.

The application code typically run on an edge device, near the camera, but might execute directly on a smart camera, or could run in the cloud if the full video stream is already streamed there through an advanced NVR system.  For high frame rate video (>1fps) the application should be configured to run the detectors locally using an [edge-endpoit](https://github.com/groundlight/edge-endpoint) server.  (This is recommended for any framerate higher than 1 frame/minute).
The edge-endoint server is an open-source service that acts identically to the cloud API as far as the SDK is concerned.  So the same application code can be used for local development, and for the deployed application - the only difference is changing the `GROUNDLIGHT_ENDPOINT` environment variable to point to the local server (e.g. `http://localhost:30101`) instead of its default value of `https://api.groundlight.ai`.

## Detector setup

Groundlight detectors can perform simple computer-vision tasks reliably, using a combination of fast local models, powerful cloud models, and live human oversight.  Every Groundlight detector responds to inference requests with both a prediction and a confidence score.  If the confidence score is below a configurable threshold, the default behavior is to escalate the image to the next level for more careful analysis, although this behavior can be overridden.  (See `ask_async` vs `ask_ml` vs `ask_confident` in the SDK guide page `submitting-image-queries`)

Groundlight detectors can be of the following types:
- Binary classification: Answer YES / UNCLEAR / NO.  These are the most reliable and easy to understand detectors.
- Multiclass classification: Answer one of K classes.  (Best with k <= 6)
- Counting: Count the number of objects in an image.  Also returns their locations as bounding boxes.

Some but not all accounts also have access to the following types:
- Text recognition: Detect and transcribe text in an image.
- Object detection: Detect and localize multiple objects in an image.  (Functionally similar to counting, but optimizes quality for bounding boxes, not the correct count.  Cannot distinguishi different classes of objects.)

Note that counting and object-detection models (collectively "OD" models) cost about 10x more than the binary-classification models for inference.  As such a typical inference pipeline consists of three detectors:

- Binary classification: Does the image have something that deserves a closer look?
- Counting: Find the objects of interest within the image.  Zoom in on them.
- Multiclass classification: Looking at the zoomed-in object, is it actually doing what we're interested in?  Or was there a false positive initially?  Also differentiate between different kinds of objects / activities of interest.

The first detector is typically run in "high recall" mode, where UNCLEAR's are treated like YES and passed to the next stage.  OD models don't have as straightforward control over precision/recall.

Simple tasks can be accomplished with a single binary or multiclass detector.  But the 3-detector pipeline is the most reliable.

## Practicalities

Configure a grabber object using the `framegrab` library from the cameras.yaml file.  If you're authoring a full demo, author this file as well.  Leave secrets like RTSP login credentials.

Assume the `GROUNDLIGHT_API_TOKEN` and `GROUNDLIGHT_ENDPOINT` environment variables are already set when the application starts.  Simply instantiating the Groundlight SDK client will use these automatically - no need to configure them in code.










