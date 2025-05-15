# How to Build a Groundlight Application

Groundlight applications are Agentic processes that monitor video streams in the background for you, to monitor for specified events or conditions.  

## Architecture

The components to a full system are:
- A camera
- The Groundlight application code
- The Groundlight Detectors (CV / ML models)
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

Note that counting and object-detection models (collectively "OD" models) cost about 10x more than the binary-classification models for inference.  

When an AI agent is building a groundlight application, it should create the detectors using the MCP server capabilities, instead of writing code to call the Groundlight API to configure the detectors.

## Coding the Applications

Applications are coded in python, and run from the command line without arguments.

### Demo Applications

A simple or demo groundlight application typically consists of a single binary or multiclass detector.  These are the most reliable and easy to understand detectors.  For quick demos, these are a great choice.

Demo applications should include plenty of print statements, and use the imgcat library to preview images as they're fetched.

### Advanced Applications

An "advanced" groundlight application typically consists of a 3-detector pipeline:

- Binary classification: Does the image have something that deserves a closer look?
- Counting: Find the objects of interest within the image.  Zoom in on them.
- Multiclass classification: Looking at the zoomed-in object, is it actually doing what we're interested in?  Or was there a false positive initially?  Also differentiate between different kinds of objects / activities of interest.

The first detector is typically run in "high recall" mode, where UNCLEAR's are treated like YES and passed to the next stage.  OD models don't have as straightforward control over precision/recall.

Simple or demonstration tasks can be accomplished with a single binary or multiclass detector.  But the 3-detector pipeline is the most reliable.

## Practicalities

Configure a grabber object using the `framegrab` library from the cameras.yaml file.  If you're authoring a full demo, author this file as well.  Leave secrets like RTSP login credentials.  In most cases, the grabber should be configured with simple motion detection, to reduce inference workloads when the scene is still.

For demo purposes, this is a super-generic cameras.yaml that will probably "just work" for a laptop's built-in webcam:

```
image_sources:
- input_type: generic_usb
  name: generic_usb_unnamed_0
  options: {}
  serial_number: null
```

Customers should be advised they can create their own by running:

```
framegrab autodiscover --preview none > cameras.yaml
```

and then test it / iterate with

```
framegrab preview cameras.yaml
```

Assume the `GROUNDLIGHT_API_TOKEN` and `GROUNDLIGHT_ENDPOINT` environment variables are already set when the application starts.  Simply instantiating the Groundlight SDK client will use these automatically - no need to configure them in code.





# Sample code

## Demonstration app with Voice Alert.

Here is an example of a complete demonstration app, which plays a voice alert when the detector
state changes from YES to NO.  Note the detector_id is hard-coded into the application. 
This is because the Agent should use the Groundlight MCP server to create the detector before authoring the app code.


```
#!/usr/bin/env python3
import os
import time
import traceback
import tempfile
import platform

from groundlight import Groundlight
from imgcat import imgcat
import cv2
import framegrab
from gtts import gTTS


class VoiceAlerts():
    """Plays sounds of a TTS voice reading messages.
    """

    def __init__(self, message:str="voice alert"):
        self.message = message
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sound_path = os.path.join(self.temp_dir.name, "alert.mp3")
        self._generate_sound()
        self.player = "afplay" if platform.system() == "Darwin" else "mpg321"

    def _generate_sound(self):
        """Generates the TTS sound file from the message."""
        tts = gTTS(text=self.message, lang='en')
        tts.save(self.sound_path)

    def alert(self):
        """Plays a sound."""
        print(f"Playing voice alert: {self.message}")
        os.system(f"{self.player} {self.sound_path}")

    def __del__(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()


class SimpleApp():
    """A simple detector app which plays a voice alert that says "You did the thing!"
    When the detector changes from YES state to NO state.
    """

    def __init__(self):
        all_cameras = framegrab.FrameGrabber.from_yaml("./cameras.yaml")
        self.camera = all_cameras[0]  # Use the first camera
        self.motdet = framegrab.MotionDetector(pct_threshold=0.5, val_threshold=50)
        self.gl = Groundlight()
        detector_id = os.environ.get("DETECTOR_ID", "det_2x8hkwPnOp8EGJLgZBdCF54I2ez")
        self.detector = self.gl.get_detector(detector_id)
        print(f"Using {self.detector}")
        # Timing constants
        self.exception_backoff_s = 60
        self.motion_interval_s = 2
        self.no_motion_post_anyway_s = 7200
        self.last_motion_post = time.time() - self.no_motion_post_anyway_s
        self.noisy = VoiceAlerts(message="You did the thing!")

        self.last_state = "?"

    def run(self):

        while True:
            try:
                self.process_frame()
            except Exception as e:
                traceback.print_exc()
                time.sleep(self.exception_backoff_s)
    
    def process_frame(self):
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        big_img = self.camera.grab()  # big_img is a numpy array
        img = cv2.resize(big_img, (800, 600))  # smaller for preview and motdet
        motion_detected = self.motdet.motion_detected(img)
        if not motion_detected:
            print(f"no motion at {now}")
            if time.time() - self.last_motion_post > self.no_motion_post_anyway_s:
                print(f"No motion detected for {self.no_motion_post_anyway_s}s, posting anyway")
                motion_detected = True
        if motion_detected:
            self.last_motion_post = time.time()
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgcat(rgb_img)
            img_query = self.gl.ask_ml(detector=self.detector, image=big_img)
            if img_query.result.label == "YES":
                print(f"YES at {now} iqid={img_query.id}")
                self.last_state = "YES"
            elif img_query.result.label == "NO":
                print(f"NO at {now} iqid={img_query.id}")
                if self.last_state == "YES":
                    self.noisy.alert()
                self.last_state = "NO"
        time.sleep(self.motion_interval_s)

if __name__ == "__main__":
    SimpleApp().run()
```

