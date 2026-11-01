EduMotion — Claude Development Prompt

Build a working Emotion-Aware Learning website. Focus only on development, integration, reliability and simple implementation.

1. Inspect First

Use graphify-out to read/inspect the existing repository files before coding.

Locate and inspect emotion_cnn.pkl.

Find its real preprocessing, input shape, labels, normalization and serialization.

Inspect existing cloud emotion API code and use its actual request/response format.

Run the current project if possible.

Reuse working code. Do not rewrite unnecessarily.

Never guess model preprocessing or API contracts.

2. Stack

Preferred: Python + FastAPI + Uvicorn, OpenCV, NumPy, HTTPX, HTML/CSS/Vanilla JS, .env.
Do not add React, database, authentication, Docker or other unnecessary infrastructure.

3. Structure

backend/
  main.py
  emotion_service.py
  cloud_emotion.py
  local_emotion.py
  image_service.py
frontend/
  index.html
  learning.html
  style.css
  app.js
  camera.js
  emotion.js
emotion_cnn.pkl
.env
.env.example
requirements.txt
README.md

Keep modules small.

4. Home Page

Create a professional EduMotion page with project introduction, workflow, technologies, editable team members and a Start Learning button. Keep project/team information in one frontend configuration object.

5. Learning Dashboard

Create a dashboard similar to the supplied EduMotion screenshots. Include course/lesson title, lesson text, configurable YouTube video, More Videos, Next Lesson, course progress and a right-side Emotion Monitor.

6. Webcam

Use navigator.mediaDevices.getUserMedia(). Show a black webcam rectangle with:
Start Camera, Stop Camera, Start Auto Detection, Stop Processing.
Camera and processing are separate. Camera ON must not start processing. Stop Processing keeps camera ON. Stop Camera stops all media tracks.

7. Core Flow

Webcam → JPEG/Base64 → POST /api/emotion/predict-base64
→ FastAPI → Cloud API OR Local CNN → Emotion + Confidence
→ Stable State → Adaptive UI

Process only every 1–2 seconds (default 1500 ms). Never process every video frame. Prevent concurrent inference requests.

8. Inference Modes

Support cloud, local, auto.

cloud = cloud API.

local = emotion_cnn.pkl.

auto = cloud first, local fallback if cloud fails.
Keep selection inside emotion_service.py.
.env:

EMOTION_MODE=cloud
EMOTION_API_URL=
EMOTION_API_KEY=
PROCESSING_INTERVAL_MS=1500
MIN_CONFIDENCE=0.55
SMOOTHING_WINDOW=5

Never expose the API key to frontend code.

9. Model Adapters

cloud_emotion.py: only cloud communication; handle timeout/network/HTTP/invalid responses; normalize the real response.
local_emotion.py: load emotion_cnn.pkl using the correct method and actual preprocessing; expose predict_emotion(image).
image_service.py: decode Base64, convert to OpenCV/NumPy, validate image, apply only required preprocessing.

10. Backend API

Implement only:

GET /api/health
POST /api/emotion/predict-base64
GET /api/emotion/status

Request:

{"image":"<base64-image>"}

Response:

{"emotion":"happy","learning_state":"happy","confidence":0.87,"source":"cloud"}

Frontend communicates only with FastAPI.

11. Emotion States

Support:
Calm, Stressed, Happy, Focused, Tired, Confused.
If raw model output is standard emotions, initially map:
neutral→Calm, happy→Happy, sad→Tired, angry→Stressed, fear→Stressed, disgust→Stressed, surprise→Confused.
Keep mapping configurable. If the model already outputs the six states, preserve them.

12. Confidence + Smoothing

Use MIN_CONFIDENCE=0.55 and a rolling history of 5 predictions. Do not change the UI from one noisy prediction. If confidence is too low, keep the previous stable state and show Low confidence — maintaining previous state.

13. Manual Mode

Allow selection of all six states. Manual selection immediately applies its UI and shows Manual Mode. Starting Auto Detection switches to Auto Mode.

14. Adaptive UI

Do not only change colors:

Calm → normal interface.

Happy → positive feedback.

Focused → reduce distractions.

Stressed → reduce visual complexity + offer break/simpler explanation.

Tired → reduce content + suggest short break.

Confused → Help card with Simplify, Show Example, Replay Video.
Support:
Happy+Focused → positive + focused
Confused+Stressed → simplify + example + break
Tired+Confused → reduce content + summary + break
Stressed+Tired → reduce visual load + break
Keep combination rules in one configuration object and normalize ordering.

15. Error Handling

Never crash for camera denied/unavailable, no face, API timeout/failure, invalid API response, local model failure, invalid image, backend unavailable or low confidence. Show simple messages. No-face should keep the previous stable state.

16. Requirements

Start with:

fastapi
uvicorn
opencv-python
numpy
httpx
python-dotenv

Add joblib, TensorFlow, PyTorch, scikit-learn etc. only if inspection proves they are required.

17. Development Order

1. graphify-out → inspect files
2. Run/understand current project
3. Verify emotion_cnn.pkl
4. Verify cloud API
5. Make FastAPI run
6. Implement cloud/local/auto service
7. Implement camera start/stop
8. Send one webcam frame to backend
9. Add periodic processing
10. Add confidence + smoothing
11. Add manual states
12. Add adaptive UI
13. Add Home page
14. Add YouTube
15. Test failures
16. Verify complete flow

Do not polish UI before the AI pipeline works.

18. Acceptance Test

The final application must perform:

Home → Start Learning → Dashboard → Start Camera → Live Webcam
→ Start Auto Detection → Capture Image → FastAPI
→ Cloud/Local CNN → Emotion → Confidence/Smoothing
→ Learning State → UI Adaptation

Also verify:
Manual Confused→Help UI, Manual Focused→Focus UI, Manual Stressed→Reduced UI, Stop Processing→Camera stays ON, Stop Camera→camera stops, Cloud failure + auto→Local fallback.
Build the smallest reliable implementation. Do not add unrelated features.




<!-- EduMotion — Claude Development Specification

Purpose

Build the working EduMotion application from the existing project assets.

This is an implementation specification, not a project report, presentation, or theory document.

The application is an emotion-aware learning website with:

Home page

Learning Dashboard

YouTube educational content

Browser webcam access

Start/Stop camera controls

Start/Stop emotion processing

Manual emotion selection

Cloud emotion API integration

Local emotion_cnn.pkl fallback

Confidence handling

Temporal smoothing

Adaptive learning UI

Priority:

FUNCTIONALITY → CORRECT INTEGRATION → RELIABILITY → CLEAN UI

Do not add unnecessary features before the core system works.

1. Existing Assets

The project already has:

emotion_cnn.pkl

This is the local trained emotion model.

There is also an existing cloud AI model/API. The cloud API accepts an image address/path/URL and returns an emotion prediction.

Do NOT guess the cloud API URL, request format, authentication, or response format.

Create an isolated cloud API adapter so the actual API contract can be inserted without changing the rest of the application.

2. Instructions for Claude

Before changing code:

Inspect the complete repository.

Identify the current frontend and backend.

Locate emotion_cnn.pkl.

Determine how the model was serialized.

Inspect any existing model-training/preprocessing code.

Inspect existing cloud API integration/configuration.

Run the current application if possible.

Preserve useful existing code.

Do not rewrite working components unnecessarily.

Implement incrementally and test each major component.

If something is unknown, isolate it behind configuration or an adapter instead of inventing behavior.

3. Technology Stack

Backend

Use:

Python

FastAPI

Uvicorn

Pydantic

HTTPX

OpenCV

NumPy

python-dotenv

joblib or pickle as required by the actual model

Only add TensorFlow, PyTorch, scikit-learn, or other ML frameworks if the actual emotion_cnn.pkl requires them.

Frontend

Use:

HTML5

CSS3

Vanilla JavaScript

Do not introduce React/Next.js/Vue unless the existing project already uses one.

4. Project Structure

Use or adapt:

edumotion/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   │
│   ├── routes/
│   │   ├── emotion_routes.py
│   │   ├── lesson_routes.py
│   │   └── system_routes.py
│   │
│   ├── services/
│   │   ├── emotion_service.py
│   │   ├── cloud_emotion_service.py
│   │   ├── local_emotion_service.py
│   │   ├── image_service.py
│   │   └── adaptation_service.py
│   │
│   ├── models/
│   │   ├── schemas.py
│   │   └── emotion_cnn.pkl
│   │
│   └── utils/
│       └── logger.py
│
├── frontend/
│   ├── index.html
│   ├── learning.html
│   ├── css/
│   │   ├── style.css
│   │   ├── home.css
│   │   └── learning.css
│   ├── js/
│   │   ├── config.js
│   │   ├── app.js
│   │   ├── camera.js
│   │   ├── emotion.js
│   │   ├── adaptation.js
│   │   └── learning.js
│   └── assets/
│
├── tests/
│   ├── test_health.py
│   ├── test_emotion.py
│   └── test_adaptation.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

Simplify this structure if the existing project already has a good organization.

5. Core Architecture

The working data flow must be:

Browser Webcam
      ↓
Camera JavaScript
      ↓
Capture JPEG Frame
      ↓
POST /api/emotion/predict-base64
      ↓
FastAPI
      ↓
Emotion Service
      ↓
 ┌─────────────────┐
 │ Cloud API       │
 │       OR        │
 │ Local CNN       │
 └─────────────────┘
      ↓
Normalized Prediction
      ↓
Confidence Filter
      ↓
Temporal Smoothing
      ↓
Learning-State Mapping
      ↓
Adaptive UI Decision
      ↓
Frontend Update

The frontend must never directly expose or call the cloud API with its credentials.

6. Configuration

Create .env.example:

APP_ENV=development
HOST=127.0.0.1
PORT=8000

EMOTION_MODE=cloud

EMOTION_API_URL=
EMOTION_API_KEY=
EMOTION_API_METHOD=POST
EMOTION_API_IMAGE_FIELD=image

PROCESSING_INTERVAL_MS=1500
MIN_CONFIDENCE=0.55
SMOOTHING_WINDOW=5
MIN_STABLE_PREDICTIONS=3

DEMO_MODE=false
DEBUG_MODE=false

.env must be in .gitignore.

Never expose EMOTION_API_KEY to frontend JavaScript.

7. Frontend Configuration

Create:

frontend/js/config.js

Example:

const APP_CONFIG = {
    projectName: "EduMotion",

    courseName: "Introduction to Machine Learning",

    lessonTitle: "Introduction to Machine Learning",

    lessonDescription:
        "Overview of machine learning and its real-world applications.",

    youtubeVideoUrl: "",

    courseProgress: 40,

    teamMembers: [
        {
            name: "Member 1",
            role: "AI/ML"
        },
        {
            name: "Member 2",
            role: "Backend"
        }
    ]
};

All project-specific text should be easy to change here.

Do not hardcode team names throughout HTML.

8. Home Page

Create:

frontend/index.html

Only include the required sections:

Navigation

Hero

Project explanation

How the system works

Technologies

Team members

Footer

Hero:

Learning That Understands You

Description:

An AI-powered learning environment that detects learner interaction states and adapts the learning experience dynamically.

Button:

Start Learning

It must open:

learning.html

9. Learning Dashboard

Create:

frontend/learning.html

Use a two-column desktop layout.

Left/Center

Learning content.

Right

Emotion monitoring panel.

10. Learning Content

Show:

Introduction to Machine Learning

Lesson:

Machine Learning

Machine learning is a field of computer science that uses statistical
techniques to give computer systems the ability to learn from data
without being explicitly programmed.

Buttons:

More Videos
Next Lesson

Keep lesson information configurable.

11. YouTube Integration

Use:

APP_CONFIG.youtubeVideoUrl

Do not hardcode a YouTube URL in multiple places.

If it is empty, display:

Video not configured

rather than a broken iframe.

12. Camera

Use:

navigator.mediaDevices.getUserMedia()

Do not start the camera automatically.

Initial state:

Camera inactive

Button:

Start Camera

When started:

Display live video.

Update camera status.

Change button to Stop Camera.

When stopped:

Stop all media tracks.

Clear the video element.

Display Camera inactive.

13. Camera and Processing Are Separate

Required states:

Camera OFF + Processing OFF
Camera ON  + Processing OFF
Camera ON  + Processing ON

Starting the camera must never automatically start AI processing.

14. Automatic Emotion Processing

When the user clicks:

Start Auto Detection

perform:

Verify camera is active.

Capture a frame from the video.

Draw it to a hidden canvas.

Convert it to JPEG.

Convert JPEG to Base64.

Send it to FastAPI.

Receive normalized prediction.

Apply/update stable emotion.

Update the dashboard.

Repeat after the configured interval.

Default:

1500 ms

Do not process every webcam frame.

15. Prevent Concurrent Requests

Maintain:

isProcessingRequest

If a request is already running, do not start another.

This prevents multiple API calls from accumulating when inference is slow.

16. Stop Processing

Stop Processing must:

Stop the processing timer.

Prevent new inference requests.

Keep the camera running.

Display:

Processing stopped

Stopping processing must not stop the camera.

17. Backend Emotion Endpoint

Implement:

POST /api/emotion/predict-base64

Request:

{
    "session_id": "abc123",
    "image": "<base64 JPEG>"
}

Response:

{
    "raw_emotion": "happy",
    "learning_state": "happy",
    "confidence": 0.87,
    "source": "cloud",
    "timestamp": "2026-09-08T00:00:00",
    "processing_time_ms": 145
}

Use Pydantic request/response schemas.

18. Image Service

Create:

image_service.py

Responsibilities:

Validate Base64.

Decode Base64.

Convert bytes to NumPy.

Convert to OpenCV image.

Validate dimensions.

Reject invalid/oversized images.

Apply the preprocessing required by the actual model/API.

Do not permanently store webcam frames.

Do not invent preprocessing.

19. Cloud Emotion Service

Create:

cloud_emotion_service.py

It must provide a clean interface such as:

predict(image) -> normalized_prediction

Configuration:

EMOTION_API_URL=
EMOTION_API_KEY=
EMOTION_API_METHOD=POST
EMOTION_API_IMAGE_FIELD=image

Do not guess the API contract.

Once the actual API format is available, adapt only this module.

Handle:

Timeout

HTTP errors

Authentication failure

Rate limiting

Network failure

Invalid JSON

Missing emotion

Missing confidence

Use a finite timeout and limited retries.

Never endlessly retry.

20. Local CNN Service

Create:

local_emotion_service.py

Before implementing it:

Inspect emotion_cnn.pkl and determine the actual serialization and input requirements.

Possible loading methods:

joblib.load(...)

or:

pickle.load(...)

Determine:

Model type

Expected image size

Number of channels

Grayscale/RGB

Normalization

Input shape

Output format

Label mapping

Use the actual training/preprocessing code if available.

Do not invent missing preprocessing.

Interface:

predict(image) -> normalized_prediction

Return:

emotion
confidence

21. Unified Emotion Service

Create:

emotion_service.py

The API route calls only this service.

Supported modes:

cloud

Use cloud API only.

local

Use local CNN only.

auto

Cloud API
   ↓
failure
   ↓
Local CNN

The fallback must happen inside the service, not inside the route.

22. Emotion Normalization

Normalize raw model output.

Examples:

"HAPPY" → "happy"
"Happiness" → "happy"
"Neutral" → "neutral"

Canonical raw emotions:

happy
sad
angry
fear
disgust
surprise
neutral

If the actual model produces the application's six states directly, preserve them.

Unknown labels must be handled explicitly.

23. Learning-State Mapping

Application states:

calm
stressed
happy
focused
tired
confused

Initial mapping:

Raw Emotion

Learning State

neutral

calm

happy

happy

sad

tired

angry

stressed

fear

stressed

disgust

stressed

surprise

confused

Make the mapping configurable.

Do not present these states as medical diagnoses.

Use terms such as:

Detected State

or:

Current Interaction State

24. Confidence Filtering

Default:

MIN_CONFIDENCE=0.55

If confidence is below the threshold:

Do not replace the stable state.

Return the low-confidence status.

Keep the previous stable UI state.

Example:

Low confidence — maintaining previous state

25. Temporal Smoothing

Do not change the UI because of one prediction.

Maintain a rolling window:

last 5 predictions

Example:

happy
happy
neutral
happy
happy

Stable result:

happy

Configuration:

SMOOTHING_WINDOW=5
MIN_STABLE_PREDICTIONS=3

Use one authoritative smoothing implementation, preferably backend-side if session state is maintained there.

26. Session State

Do not use one global emotion history for every user.

Use a session_id supplied by the frontend.

For a local academic prototype, an in-memory dictionary is acceptable:

sessions = {
    "abc123": {
        "history": [...]
    }
}

Document that production deployment should use a persistent state store.

27. Manual Mode

The dashboard must provide:

Calm
Stressed
Happy
Focused
Tired
Confused

When the user selects one:

mode = manual

Automatic processing must not overwrite the manual state.

Display:

Manual Mode

Manual selection must immediately trigger the corresponding UI adaptation.

28. Automatic Mode

When:

Start Auto Detection

is clicked:

mode = auto

The next stable prediction can update the UI.

Display:

Auto Detection Active

29. Adaptive UI Engine

Create:

adaptation.js

and optionally:

adaptation_service.py

Do not scatter adaptation logic across HTML.

The adaptation engine should return a structured UI state, for example:

{
    mode: "help",
    message: "Would you like a simpler explanation?",
    showSimplify: true,
    showExample: true,
    showReplay: true,
    suggestBreak: false
}

30. Calm

State:

calm

UI:

Normal lesson layout

Normal content density

Standard video

Normal navigation

Message:

You're in a steady learning state.

31. Happy

State:

happy

UI:

Positive feedback

Slightly engaging presentation

Optional additional example

Message:

Great! You're engaged with the lesson.

Do not add distracting animations.

32. Focused

State:

focused

UI:

Keep lesson content prominent

Reduce unnecessary controls

Reduce distractions

Message:

You're focused. Keep going.

33. Stressed

State:

stressed

UI:

Reduce visual complexity

Reduce information density

Avoid excessive animation

Offer break

Offer simpler explanation

Message:

Take a moment. You can review this concept at your own pace.

Buttons:

Take a Short Break
Show Simpler Explanation

34. Tired

State:

tired

UI:

Reduce content density

Show concise summary

Suggest break

Avoid overwhelming options

Message:

You may want to take a short break before continuing.

35. Confused

State:

confused

UI:

Show help card

Offer simplified explanation

Show example

Offer video replay

Highlight important concepts

Message:

Would you like a simpler explanation?

Buttons:

Simplify
Show Example
Replay Video

36. Emotion Combinations

Support combinations through configuration, not large nested if/else blocks.

Normalize the combination key so:

happy+focused

and:

focused+happy

are identical.

Support at least:

happy + focused
focused + calm
confused + stressed
tired + confused
stressed + tired
confused + neutral

Recommended behavior:

happy + focused

Maintain the lesson and show positive reinforcement.

focused + calm

Reduce distractions and preserve the current learning flow.

confused + stressed

High-assistance mode:

Simplify content

Reduce information density

Show example

Offer help

Offer break

tired + confused

Reduced cognitive load:

Show summary

Suggest break

Offer simplified explanation

stressed + tired

Break mode:

Reduce visual load

Suggest break

Avoid forcing the next lesson

confused + neutral

Show assistance/simplification options.

37. Adaptation Priority

Use:

confused + stressed
stressed
tired + confused
tired
confused
focused
happy
calm
neutral

But confidence and temporal stability must always be respected.

38. Emotion Monitor UI

Right-side panel:

Emotion Monitor

Camera rectangle.

Controls:

Start Camera
Stop Camera

Start Auto Detection
Stop Processing

State:

Current State

🙂 Happy

Confidence
87%

Mode
AUTO

Recent predictions:

Happy
Happy
Calm
Happy
Focused

39. Status Indicators

Display:

Camera: Inactive

or:

Camera: Active

And:

Processing: Stopped

or:

Processing: Active

Optionally:

Emotion Service: Connected

40. Demo Mode

Add:

DEMO_MODE=false

When true, the application must remain demonstrable without a working camera, cloud API, or local model.

Manual state selection must always work.

Clearly display:

Demo Mode

Never present simulated predictions as real predictions.

41. Debug Mode

Add:

DEBUG_MODE=false

When enabled, show:

Inference source
Raw emotion
Learning state
Confidence
Processing time
API status
Camera status
Processing status

Do not expose secrets.

Hide this panel when disabled.

42. Error Handling

Handle gracefully:

Camera

Permission denied

Camera unavailable

Camera already in use

Browser unsupported

Message:

Unable to access the camera. Please check browser permissions.

No face

No face detected. Please position your face inside the camera area.

Keep the previous stable state.

Cloud API

Timeout

401/403

404

429

500+

Network failure

Invalid response

Local model

Missing model

Incorrect serialization

Missing dependency

Wrong input shape

Prediction failure

Backend

If backend is unreachable:

Unable to connect to the emotion service.

The learning page must remain usable.

43. Security

Never expose:

EMOTION_API_KEY

to the frontend.

Correct:

Browser
  ↓
FastAPI
  ↓
Cloud API

Do not:

Store webcam frames permanently

Log raw images

Log API credentials

Put credentials in JavaScript

44. Backend Routes

Implement:

GET /
GET /api/health

POST /api/emotion/predict-base64

GET /api/emotion/status

POST /api/emotion/mode

GET /api/lesson

POST /api/adaptation/evaluate

Keep route handlers thin.

Business logic belongs in services.

45. Health Endpoint

Implement:

GET /api/health

Response:

{
    "status": "ok",
    "emotion_mode": "cloud"
}

Optionally include:

local_model_available
cloud_api_configured

Never expose secrets.

46. Emotion Status Endpoint

Implement:

GET /api/emotion/status

Response:

{
    "mode": "cloud",
    "cloud_configured": true,
    "local_model_available": true
}

47. Frontend Module Responsibilities

camera.js

Only:

Start camera

Stop camera

Capture frame

Camera state

emotion.js

Only:

Start processing

Stop processing

Send frame

Receive prediction

Manage processing status

Handle inference errors

adaptation.js

Only:

Determine adaptation

Handle combinations

Update adaptive UI

learning.js

Only:

Lesson content

Progress

YouTube

Lesson navigation

app.js

Only:

Application initialization

Shared state

Configuration

48. Backend Module Responsibilities

main.py

Only:

Create FastAPI app

Register routes

Configure middleware

Serve frontend if appropriate

Do not put model logic here.

emotion_routes.py

Validate request

Call emotion service

Return response

emotion_service.py

Select inference mode

Handle cloud/local fallback

Normalize result

cloud_emotion_service.py

Cloud API communication only

local_emotion_service.py

Local CNN inference only

image_service.py

Image decoding

Validation

Required preprocessing

adaptation_service.py

Learning-state adaptation rules if backend-side adaptation is used

49. Serving the Application

Prefer serving the frontend through FastAPI during development.

The goal should be:

uvicorn backend.main:app --reload

Then open:

http://127.0.0.1:8000/

If the existing architecture requires separate frontend serving, document the exact commands.

Keep startup simple.

50. Requirements

Generate requirements.txt from actual imports.

Start with only what is required:

fastapi
uvicorn
httpx
python-dotenv
opencv-python
numpy
pydantic
joblib

Add other dependencies only when the actual project requires them.

51. Testing

Create tests for:

Health endpoint

Invalid Base64

Image decoding

Emotion normalization

Raw → learning-state mapping

Confidence filtering

Temporal smoothing

Combination normalization

Adaptation rules

Manual mode

Automatic mode

Cloud failure fallback

Tests must not require a physical webcam.

52. Development Sequence

Implement in exactly this order.

Phase 1 — Inspect

Inspect repository

Inspect frontend

Inspect backend

Inspect emotion_cnn.pkl

Inspect existing API integration

Run application

Phase 2 — Backend Foundation

Implement:

FastAPI
configuration
health endpoint
error handling

Verify startup.

Phase 3 — Local Model

Implement and test:

model loading
preprocessing
prediction
normalization

Phase 4 — Cloud Adapter

Implement the actual cloud API contract once known.

Test independently.

Phase 5 — Unified Emotion Service

Test:

cloud
local
auto
fallback

Phase 6 — Camera

Implement and verify:

Start Camera
Stop Camera

before adding AI processing.

Phase 7 — Single Prediction

Implement:

frame
→ JPEG
→ Base64
→ backend
→ model/API
→ response

Phase 8 — Automatic Processing

Add:

processing interval
request locking
start/stop

Phase 9 — Stable State

Add:

confidence
history
smoothing
stable state

Phase 10 — Manual Mode

Verify all six states.

Phase 11 — Adaptive UI

Implement each state, then combinations.

Phase 12 — Home Page

Implement editable project/team information.

Phase 13 — YouTube + Lesson

Add configurable content.

Phase 14 — Error Handling

Test failure cases.

Phase 15 — Demo Mode

Verify demonstration without external dependencies.

Phase 16 — Final Verification

Run:

backend tests
manual browser test
camera test
cloud test
local model test
fallback test
manual emotion test
adaptive UI test

Only after functionality works should visual polish be added.

53. Acceptance Criteria

The application is complete only when:

Home

Loads correctly

Team members can be edited from configuration

Project information can be edited from configuration

Start Learning works

Dashboard

Loads correctly

YouTube works when configured

Camera starts

Camera stops

Webcam appears inside the camera panel

Processing

Auto detection starts

Frames are captured

Backend receives frames

Cloud inference works when configured

Local inference works when configured

Auto fallback works

Processing stops independently from camera

UI receives predictions

Emotion

Confidence is displayed

Low-confidence results do not destabilize the UI

Temporal smoothing works

Stable state is displayed

Manual Mode

All six states work:

Calm
Stressed
Happy
Focused
Tired
Confused

Adaptation

Each state changes the interface appropriately.

Combination rules work.

Reliability

The application does not crash when:

Camera is denied

No face is detected

API fails

Model is unavailable

Invalid image is submitted

Network is unavailable

Security

API credentials remain backend-only

.env is ignored

Webcam frames are not permanently stored

Raw images are not logged

54. Final Runtime Flow

Open EduMotion
      ↓
Home
      ↓
Start Learning
      ↓
Learning Dashboard
      ↓
Start Camera
      ↓
Live Webcam
      ↓
Start Auto Detection
      ↓
Capture Frame
      ↓
FastAPI
      ↓
Cloud API / Local CNN
      ↓
Raw Emotion
      ↓
Confidence Check
      ↓
Temporal Smoothing
      ↓
Learning State
      ↓
Adaptive UI
      ↓
Continue Learning

Manual flow:

Learning Dashboard
      ↓
Select Confused
      ↓
Manual Mode
      ↓
Help UI appears
      ↓
Select Focused
      ↓
Focused UI appears
      ↓
Start Auto Detection
      ↓
Automatic Mode resumes

55. Critical Constraints

Do NOT:

Fake production API responses

Guess the cloud API contract

Guess emotion_cnn.pkl preprocessing

Expose API keys

Process webcam at maximum FPS

Change the UI for every single noisy prediction

Put all backend code in main.py

Put all frontend logic in one JS file

Store webcam images unnecessarily

Add unnecessary frameworks

Build only a frontend mockup

DO:

Inspect first

Build incrementally

Test modules independently

Use adapters

Isolate model integration

Use configuration

Use confidence thresholds

Use temporal smoothing

Provide manual mode

Provide demo mode

Handle failures

Keep startup simple

Verify actual functionality

56. Final Instruction

Start by inspecting the existing repository and assets.

Do not immediately rewrite the project.

Determine:

Current project structure

Existing backend

Existing frontend

Actual emotion_cnn.pkl format

Existing model preprocessing

Existing cloud API integration

Existing dependencies

Then implement the application according to this specification.

After each major phase, verify it works before continuing.

If cloud API details are missing, implement a clean adapter with configuration placeholders and continue building/testing the rest of the system.

If the local model preprocessing requirements are missing, inspect the model/training code and isolate the missing requirements instead of fabricating them.

The final result must be a working, runnable EduMotion application.

Priority:

1. Correctness
2. Integration
3. Reliability
4. Error handling
5. Maintainability
6. UI polish

Do not spend time on reports, presentations, theory, or unnecessary features until the core application works.
















<!-- EduMotion — AI-Powered Emotion-Aware Learning Platform



You are a senior full-stack Python developer, ML integration engineer, UI/UX designer, and software architect.

Build a complete, production-quality prototype of an AI-powered Emotion-Aware Learning Platform called EduMotion.

Important: Do not create only a visual mockup. Build a WORKING full-stack application with a clean frontend, backend, API integration, webcam processing, emotion-state management, and adaptive UI behavior.

The project is an academic/research prototype for an Emotion-Aware Dynamic UI/UX System. The core idea is:

Webcam → Face/Image Capture → Emotion Detection Model/API → Emotion Interpretation → Learning-State Decision → Adaptive UI/UX

The system should detect the learner's facial emotional state while they are watching/learning from educational content and dynamically adapt the learning interface.

1. Existing Project Context

I already have:

1. Local emotion classification model

emotion_cnn.pkl

2. Cloud-based emotion detection model/API

The cloud API accepts an address/URL/path of an image and returns the predicted emotion.

Do NOT invent the cloud API URL or request format.

Create a single configuration file such as:

backend/config.py

or:

.env

where I can easily enter:

EMOTION_API_URL=
EMOTION_API_KEY=
EMOTION_API_METHOD=
EMOTION_API_IMAGE_FIELD=

The application must make it very easy to replace the API URL or request format later.

2. Model Inference Modes

The local model emotion_cnn.pkl must also be supported through a separate inference module.

The system should have three inference modes:

cloud
local
auto

Configuration:

EMOTION_MODE="cloud"

Mode: cloud

Use the cloud API.

Mode: local

Use emotion_cnn.pkl.

Mode: auto

Attempt cloud inference.

If the API fails, times out, or returns an error, automatically fall back to the local model.

Never crash the website because the emotion API is temporarily unavailable.

3. Technology Stack

Backend

Use:

Python

FastAPI

Uvicorn

Pydantic

HTTPX / requests

OpenCV

NumPy

joblib/pickle for emotion_cnn.pkl, depending on its actual serialization

python-dotenv

Frontend

Use:

HTML5

CSS3

Vanilla JavaScript

Do not introduce React/Next.js unless absolutely necessary.

Keep the project simple enough for a college student to understand and explain in a viva.

The frontend should communicate with FastAPI using REST APIs and JSON.

Use WebSocket only if it genuinely improves webcam/emotion processing. Otherwise REST polling is preferred for simplicity and reliability.

4. Project Structure

Create a clean modular structure similar to:

edumotion/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── emotion_routes.py
│   │   ├── lesson_routes.py
│   │   └── system_routes.py
│   │
│   ├── services/
│   │   ├── cloud_emotion_service.py
│   │   ├── local_emotion_service.py
│   │   ├── emotion_service.py
│   │   ├── image_service.py
│   │   └── adaptation_service.py
│   │
│   ├── models/
│   │   ├── emotion_cnn.pkl
│   │   └── schemas.py
│   │
│   └── utils/
│       ├── logger.py
│       └── helpers.py
│
├── frontend/
│   ├── index.html
│   ├── learning.html
│   │
│   ├── css/
│   │   ├── style.css
│   │   ├── home.css
│   │   └── learning.css
│   │
│   ├── js/
│   │   ├── app.js
│   │   ├── emotion.js
│   │   ├── adaptation.js
│   │   └── learning.js
│   │
│   └── assets/
│
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── run.py

If a better modular structure is required, you may improve it, but do not unnecessarily over-engineer the application.

5. Only Two Main Frontend Pages

Create exactly two primary pages.

PAGE 1

HOME / PROJECT INTRODUCTION

PAGE 2

COURSE / LEARNING DASHBOARD

Navigation:

Home
Learning Dashboard

The design should be inspired by the provided EduMotion screenshots, but do not blindly copy them.

Create a more polished, modern academic AI-learning interface.

6. Page 1 — Home Page

The Home page should explain the project professionally.

6.1 Navigation Bar

Logo:

EduMotion

Navigation:

Home
Learning Dashboard

Right side:

Emotion-Aware Learning

6.2 Hero Section

Title:

Learning That Understands You

Subtitle:

An AI-powered learning environment that detects learner emotions and dynamically adapts the learning experience.

Buttons:

Start Learning
How It Works

6.3 Project Explanation

Explain:

Traditional educational platforms provide the same interface regardless of learner state.

EduMotion uses computer vision and AI-based emotion detection to understand the learner's current state and dynamically modify the learning interface.

6.4 How the System Works

Display a visual pipeline:

Webcam
   ↓
Face Detection
   ↓
Emotion Detection
   ↓
Emotion Analysis
   ↓
Learning-State Interpretation
   ↓
Adaptive UI
   ↓
Improved Learning Experience

6.5 Technology Section

Show cards for:

Python

OpenCV

CNN

FastAPI

HTML/CSS/JavaScript

Cloud AI API

6.6 Supported Emotional States

Display:

Calm

Stressed

Happy

Focused

Tired

Confused

6.7 Team Members

The names must be extremely easy to edit.

Create one configuration section/file such as:

frontend/js/team.js

or:

team.json

Example:

[
  {
    "name": "Member 1",
    "role": "AI/ML"
  },
  {
    "name": "Member 2",
    "role": "Backend"
  }
]

I should be able to change team member names without modifying the HTML structure.

Also make the following editable from one configuration location:

Project title

Project description

Guide name

Department

Institute

Team member names

Team member roles

6.8 Footer

Include:

EduMotion
Emotion-Aware Dynamic UI/UX System

7. Page 2 — Learning Dashboard

This is the most important page.

The layout should be similar in concept to the provided screenshots.

Use a desktop-first responsive dashboard.

Main layout

LEFT / CENTER:

Learning content

RIGHT:

Emotion + engagement monitoring panel

8. Course Header

Display:

Course:
Introduction to Machine Learning

Lesson:
Introduction to Machine Learning

Description:
Overview of machine learning and its real-world applications.

Make these values configurable.

9. Educational Content

Create a lesson content card.

Example:

Machine Learning

Machine learning is a field of computer science that uses statistical techniques to give computer systems the ability to learn from data without being explicitly programmed.

Include:

Paragraph content

More Videos button

Next Lesson button

The content must be easy to replace through a JavaScript/JSON configuration.

10. YouTube Video

Embed a YouTube educational video.

IMPORTANT:

Do not hardcode the video URL in multiple places.

Create a configuration value:

YOUTUBE_VIDEO_URL = "..."

Use the YouTube embed format safely.

The video should play inside the learning page.

If autoplay is blocked by the browser, show a normal play button rather than breaking.

11. Webcam Monitoring

Create a black rectangular webcam/video area similar to the screenshot.

Initially show:

Camera inactive

Button:

Start Camera

When camera starts:

Display live webcam footage inside the rectangle.

Use browser navigator.mediaDevices.getUserMedia().

Do not upload continuous raw video unnecessarily.

Capture individual frames for emotion processing.

Controls:

START CAMERA
STOP CAMERA

Separate processing controls:

START AUTO DETECTION
STOP PROCESSING

The user must be able to use the camera without automatically sending images to the API.

Camera ON != Emotion Processing ON.

This distinction is important.

12. Emotion Processing Flow

When the user clicks:

START AUTO DETECTION

Perform:

Capture webcam frame.

Convert frame to image.

Send image to backend.

Backend sends image to configured cloud emotion API OR local CNN model.

Receive prediction.

Normalize prediction.

Apply confidence filtering.

Apply temporal smoothing.

Determine stable emotional state.

Update UI.

Repeat after a configurable interval.

Example:

PROCESSING_INTERVAL = 1500

Do not send 30/60 requests per second.

Use approximately 1–2 predictions per second.

This reduces API cost, network usage, and UI lag.

13. Emotion API Response Handling

The cloud API may return different response structures.

Create a normalization layer:

normalize_emotion_response()

It should convert different possible responses into:

{
  "emotion": "happy",
  "confidence": 0.87,
  "source": "cloud",
  "timestamp": "...",
  "processing_time_ms": 123
}

Supported canonical raw emotions:

happy
sad
angry
fear
disgust
surprise
neutral

However, the UI should map these raw model emotions into the application's six learning states:

CALM
STRESSED
HAPPY
FOCUSED
TIRED
CONFUSED

Do not assume that the model directly predicts these six states.

14. Emotion → Learning State Mapping

Create a configurable mapping.

Recommended initial mapping:

Raw Model Emotion

Learning State

neutral

CALM

happy

HAPPY

sad

TIRED

angry

STRESSED

fear

STRESSED

disgust

STRESSED

surprise

CONFUSED

If the model itself already outputs:

calm
stressed
happy
focused
tired
confused

then preserve those values directly.

Important

Emotion recognition is probabilistic.

Do not claim that facial expression proves the user's actual psychological state.

Use UI terminology such as:

Detected State

or:

Current Interaction State

rather than making medical or psychological claims.

15. Temporal Smoothing

Do not change the UI immediately whenever one frame produces a different emotion.

Example predictions:

Happy
Happy
Neutral
Happy
Happy

Stable state:

Happy

Use a rolling window:

last 5 predictions

and select the dominant/high-confidence state.

Recommended:

minimum confidence = 0.55
minimum stable predictions = 3

This prevents UI flickering.

Create:

emotion_history[]

and:

get_stable_emotion()

16. Confidence Display

The dashboard should show:

Current Emotion / Detected State

Example:

🙂 Happy

Confidence:
87%

Also show a progress bar.

If confidence is too low:

Low confidence — maintaining previous state

Do not aggressively change the interface based on uncertain predictions.

17. Emotion States and UI Adaptation

The system should not simply change the background color.

It should adapt the learning experience intelligently.

Create:

adaptation.js

and:

adaptation_service.py

18. State 1 — Calm

Meaning:

Stable learning state.

UI:

Normal lesson layout

Standard video

Standard text

Normal navigation

Normal information density

Recommendation:

You're in a steady learning state.

19. State 2 — Happy

UI:

Slightly more engaging visual presentation

Positive feedback

Optional additional example

Encourage continuation

Message:

Great! You're engaged with the lesson.

Do not make the interface overly colorful or distracting.

20. State 3 — Focused

UI:

Reduce unnecessary UI elements

Keep lesson content prominent

Minimize distractions

Keep video and important information visible

Message:

You're focused. Keep going.

21. State 4 — Stressed

UI:

Reduce visual complexity

Reduce information density

Provide a short explanation

Offer pause/break option

Avoid excessive animations

Message:

Take a moment. You can review this concept at your own pace.

Possible controls:

Take a Short Break
Show Simpler Explanation

22. State 5 — Tired

UI:

Reduce content density

Suggest a short break

Offer concise summary

Reduce unnecessary animation

Provide Continue instead of overwhelming options

Message:

You may want to take a short break before continuing.

23. State 6 — Confused

This is one of the most important adaptive states.

UI:

Show "Need Help?" card

Provide simplified explanation

Show key points

Offer example

Offer replay of current video

Highlight important terms

Message:

Would you like a simpler explanation?

Buttons:

Simplify
Show Example
Replay Video

24. Emotion Combinations

The system must support combinations because real-world emotion predictions are not always clean.

Examples:

Happy + Focused

→ Maintain current lesson
→ Show positive reinforcement
→ Do not introduce distractions

Focused + Calm

→ Deep-learning mode
→ Minimize UI distractions

Confused + Stressed

→ Highest assistance mode
→ Simplify content
→ Reduce information density
→ Offer example
→ Offer pause/help

Tired + Confused

→ Short summary
→ Reduce content
→ Suggest break
→ Then offer simplified explanation

Happy + Excited/Surprise

→ Interactive example
→ Optional additional content

Neutral + Focused

→ Standard learning mode
→ Keep current content

Stressed + Tired

→ Break recommendation
→ Reduce visual load
→ Do not force next lesson

Confused + Neutral

→ Show explanation/help option

Do not create hundreds of hardcoded if/else statements.

Create an adaptation configuration object such as:

EMOTION_COMBINATIONS = {
    "confused+stressed": {
        "mode": "high_assistance",
        "show_help": true,
        "simplify": true,
        "suggest_break": true
    },

    "tired+confused": {
        "mode": "reduced_load",
        "show_summary": true,
        "suggest_break": true
    },

    "happy+focused": {
        "mode": "focused_engagement",
        "show_positive_feedback": true
    }
}

Normalize ordering so:

focused+happy

and:

happy+focused

produce the same combination.

25. Adaptation Priority

If multiple states are detected over time, use a priority system.

Recommended priority:

Confused + Stressed

Stressed

Tired + Confused

Tired

Confused

Focused

Happy

Calm

Neutral

However, do not let one low-confidence prediction override a stable state.

Use:

confidence + temporal smoothing

26. Manual Emotion Controls

The user specifically needs manual options.

Create a panel:

Current Emotion

Options:

○ Calm
○ Stressed
○ Happy
○ Focused
○ Tired
○ Confused

And:

Start Auto Detection
Stop Processing

The user can manually select any state.

When manually selected:

MODE = MANUAL

Automatic adaptation must temporarily stop.

Display:

Manual Mode

When:

START AUTO DETECTION

is clicked:

MODE = AUTO

Display:

Auto Detection Active

This is extremely important for demonstration and viva.

27. Current State Panel

Right-side panel:

EMOTION MONITOR

Black webcam rectangle

Camera controls

Processing controls

Current Detected State

Confidence %

Detection Mode

Recent emotion history

Example:

Current State
🙂 Happy

Confidence
87%

Mode
AUTO

Recent:

Happy
Happy
Calm
Happy
Focused

28. Course Progress

Show:

Course Progress

40% Complete

with progress bar.

Make percentage configurable.

Optionally calculate it from completed lessons.

29. Engagement Tracking

Create an:

Engagement Tracking

section.

Do not falsely claim that the system scientifically measures engagement.

Instead calculate a prototype engagement indicator from:

Stable focused state

Happy state

Calm state

Time spent on lesson

Video interaction

Example:

Estimated Engagement:
72%

Clearly label this as:

Estimated Engagement

or:

Prototype Engagement Indicator

30. Backend API Design

Create endpoints:

GET /
GET /api/health

POST /api/emotion/predict
POST /api/emotion/predict-base64

GET /api/emotion/status

POST /api/emotion/mode

GET /api/lesson

POST /api/adaptation/evaluate

The most important endpoint:

POST /api/emotion/predict-base64

Request:

{
    "image": "<base64 image>"
}

Response:

{
    "emotion": "happy",
    "learning_state": "happy",
    "confidence": 0.87,
    "source": "cloud",
    "processing_time_ms": 145
}

Do not expose API keys to the frontend.

The browser must communicate only with our backend.

31. Image Processing

For webcam frames:

Frontend:

Webcam frame
↓
Canvas
↓
JPEG
↓
Base64
↓
Backend

Backend:

Base64 decode
↓
OpenCV image conversion
↓
Optional face detection/cropping
↓
Cloud API OR local CNN
↓
Prediction normalizer
↓
Response JSON

Do not store user images permanently unless explicitly configured.

Delete temporary images after processing.

32. Local CNN Model

Create:

local_emotion_service.py

Load:

emotion_cnn.pkl

ONLY use the appropriate loader based on how the actual model was serialized.

Before implementing loading logic:

inspect/verify the model type if possible.

If it is a joblib/scikit-learn model:

joblib.load()

If pickle:

pickle.load()

If the model cannot be loaded because preprocessing information is missing:

DO NOT silently invent preprocessing.

Create a clear error message describing what is missing.

The local model interface should expose:

predict_emotion(image)

returning:

emotion
confidence

33. Cloud API Service

Create:

cloud_emotion_service.py

Responsibilities:

Send image to cloud API

Add authentication

Handle timeout

Handle HTTP errors

Parse response

Normalize response

Return canonical prediction

Use:

HTTP timeout

Retry only a limited number of times.

Do not endlessly retry.

If cloud API fails and:

EMOTION_MODE=auto

use the local model.

34. Security

Never put:

EMOTION_API_KEY

inside frontend JavaScript.

Use:

.env

Add .env to .gitignore.

Create:

.env.example

with placeholders.

Validate captured image size.

Limit request size.

Do not save webcam images by default.

35. Performance

The system must remain responsive.

Do not process every webcam frame.

Camera can display at approximately 30 FPS.

Emotion inference should happen approximately every:

1–2 seconds

Use:

lastPredictionTime

and:

PROCESSING_INTERVAL

Prevent duplicate concurrent API requests.

If one inference request is still running, do not send another.

36. Failure Handling

Handle all of the following:

Camera denied

Camera unavailable

No face detected

Cloud API timeout

Cloud API returns invalid response

Cloud API unavailable

Local model missing

Model loading failure

Low confidence

Network disconnected

Backend unavailable

Malformed image

YouTube unavailable

Browser autoplay restriction

The UI should show friendly messages instead of stack traces.

Examples:

Camera permission is required for automatic detection.

Unable to connect to the emotion service. Please try again.

No face detected. Please position your face inside the camera area.

37. Logging

Backend logging should record:

Timestamp

Request

Inference source

Processing time

Prediction

Confidence

Errors

Do not log:

Raw images

API keys

Sensitive user information

38. Normalized Data Model

Create a normalized emotion object:

{
    "raw_emotion": "happy",
    "learning_state": "happy",
    "confidence": 0.87,
    "source": "cloud",
    "timestamp": "...",
    "processing_time_ms": 120
}

This object should be the common interface between model modules and the UI.

This is critical because the frontend should not care whether prediction came from:

CNN local model

or:

Cloud AI API

39. Module Interaction

Architecture:

Webcam
 ↓
Frontend Camera Module
 ↓
Image Capture
 ↓
POST /api/emotion/predict-base64
 ↓
Emotion Service
 ↓
 ┌─────────────────────┐
 │ Cloud Emotion API   │
 │        OR           │
 │ Local CNN Model     │
 └─────────────────────┘
 ↓
Prediction Normalizer
 ↓
Temporal Smoothing
 ↓
Learning-State Mapper
 ↓
Adaptation Engine
 ↓
JSON Response
 ↓
Frontend Emotion State
 ↓
Adaptive UI

Create clean interfaces between modules.

Do not put all logic inside main.py.

40. UI Design

Visual style:

Modern

Professional

Academic

Minimal

Clean

AI/education themed

Use:

Rounded cards

Subtle shadows

Clean typography

Responsive layout

Clear status indicators

Avoid:

Excessive gradients

Neon colors

Too many animations

Overcrowding

Use emotion indicators carefully.

Example:

Calm → neutral/soft presentation
Happy → positive visual feedback
Focused → distraction-free presentation
Stressed → calmer/reduced interface
Tired → simplified interface
Confused → help-oriented interface

Do not rely solely on color to communicate state.

Use:

icon + text + status

41. Accessibility

Include:

Proper labels

Keyboard-accessible buttons

ARIA labels where useful

Sufficient contrast

Visible focus states

Do not make emotion information available only through color.

42. Responsive Design

Desktop

Two-column dashboard.

Tablet

Reduced side panel.

Mobile

Stack learning content and monitoring panel vertically.

Camera area should remain usable.

43. Configuration

Create one easy-to-edit configuration system.

For example:

config.js

containing:

PROJECT_NAME
COURSE_NAME
LESSON_TITLE
LESSON_DESCRIPTION
YOUTUBE_VIDEO_URL
TEAM_MEMBERS
SUPPORTED_EMOTIONS
PROCESSING_INTERVAL
MIN_CONFIDENCE
COURSE_PROGRESS

Backend configuration should be in:

.env

This allows the project owner to modify project details without touching application logic.

44. README

Create an extremely detailed README explaining:

Project overview

Features

Architecture

Folder structure

Python installation

Virtual environment creation

Requirements installation

.env configuration

emotion_cnn.pkl placement

Cloud API configuration

Running backend

Running frontend

Camera permission

Testing cloud mode

Testing local mode

Testing auto mode

Manual emotion mode

Demo mode

Troubleshooting

Commands should be provided for Windows.

Example:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload

45. requirements.txt

Generate a complete requirements.txt based on the actual imports.

Do not include unnecessary packages.

Likely dependencies:

fastapi
uvicorn
opencv-python
numpy
pandas
python-dotenv
httpx
pydantic
joblib

Add TensorFlow/PyTorch ONLY if emotion_cnn.pkl actually requires it.

Do not install massive ML frameworks unnecessarily.

46. Testing

Create tests for:

API health

Image decoding

Emotion normalization

Emotion mapping

Confidence filtering

Temporal smoothing

Combination handling

Cloud API failure

Local model fallback

Invalid image

Manual mode

Automatic mode

Create sample mock API responses.

The project should be testable without a physical webcam.

47. Demo Mode

Create a:

DEMO MODE

This is extremely important for college presentation/viva.

If:

DEMO_MODE=true

the application should allow simulated emotion predictions:

Calm
Stressed
Happy
Focused
Tired
Confused

The manual buttons should demonstrate the adaptive UI immediately.

This means the entire project can be demonstrated even if:

Webcam does not work

API is offline

Internet is unavailable

Model is unavailable

Clearly label:

Demo Mode

Do not mix fake/demo predictions with real predictions silently.

48. Debug Panel

Create an optional developer/debug panel enabled through configuration.

Show:

Inference source:
Cloud / Local

Raw emotion:
happy

Mapped state:
Happy

Confidence:
0.87

Processing time:
145 ms

API status:
Connected

Camera:
Active

Processing:
Active

Hide it by default for normal users.

49. Viva-Friendly Architecture

The implementation must be easy to explain.

A student should be able to answer:

Why OpenCV?

For webcam frame acquisition, image processing, and optional face detection/cropping.

Why CNN?

For extracting visual facial features and classifying facial expressions.

Why cloud API?

For scalable/remote inference and the ability to use a hosted model.

Why local model?

For offline fallback, reduced dependency on network connectivity, and local inference.

Why FastAPI?

For a lightweight, fast backend API and clean separation between frontend and ML services.

Why temporal smoothing?

To prevent the interface from flickering due to one-frame prediction changes.

Why confidence threshold?

To avoid adapting the UI based on unreliable predictions.

Why manual mode?

For user control, testing, debugging, and reliable project demonstration.

Why adaptive UI?

The goal is not merely emotion recognition; the goal is to use the detected interaction state to improve the learning experience.

How does frontend communicate with backend?

Through REST API calls using JSON/Base64 image payloads.

How is API key protected?

The frontend never receives the API key; only the backend accesses the cloud API.

What happens if cloud API fails?

In auto mode, the system attempts local CNN inference.

What happens if no face is detected?

Return a friendly status and maintain the previous stable state.

Why not process every webcam frame?

It would unnecessarily increase CPU/GPU usage, network traffic, API calls, and latency.

How are multiple emotions handled?

Using confidence filtering, temporal smoothing, and configurable combination rules.

How does the system avoid UI flickering?

By maintaining a rolling history and requiring stable/high-confidence predictions before changing the UI state.

50. Important Engineering Rules

DO NOT

Hardcode API keys

Hardcode team names throughout HTML

Put all code into one file

Send webcam frames continuously at maximum FPS

Change emotion on every single prediction

Expose cloud credentials

Make unsupported psychological/medical claims

Pretend confidence equals actual emotional certainty

Break the UI when API fails

Invent the cloud API response structure

Invent preprocessing for emotion_cnn.pkl

Add unnecessary frameworks

DO

Modularize

Use configuration

Use clean APIs

Use fallback inference

Use temporal smoothing

Use confidence thresholds

Provide manual controls

Provide demo mode

Provide error handling

Provide clear documentation

Keep the architecture understandable

51. Final Expected Output

Generate the COMPLETE WORKING PROJECT.

Provide:

All source files

Backend

Frontend

CSS

JavaScript

API integration

Local CNN integration

Configuration

requirements.txt

.env.example

README

Tests

Demo mode

Error handling

After generating the project, explain:

A. What every file does

B. How the modules communicate

C. Exact data flow from webcam → model → UI

D. How cloud/local fallback works

E. How emotion combinations work

F. How adaptive UI works

G. How to run the project

H. How to replace the cloud API

I. How to replace emotion_cnn.pkl

J. How to add another emotion

K. How to add another lesson

L. How to change team members

52. Most Important Implementation Principle

The final system should behave like this:

USER OPENS EDUMOTION
        ↓
HOME PAGE
        ↓
START LEARNING
        ↓
LEARNING DASHBOARD
        ↓
YOUTUBE EDUCATIONAL CONTENT
        +
WEBCAM MONITOR
        ↓
USER CLICKS "START CAMERA"
        ↓
LIVE CAMERA APPEARS
        ↓
USER CLICKS "START AUTO DETECTION"
        ↓
FRAME CAPTURE
        ↓
BACKEND
        ↓
CLOUD API
        OR
LOCAL CNN
        ↓
RAW EMOTION
        ↓
CONFIDENCE CHECK
        ↓
TEMPORAL SMOOTHING
        ↓
LEARNING STATE
        ↓
ADAPTATION ENGINE
        ↓
UI CHANGES
        ↓
USER CONTINUES LEARNING
        ↓
SYSTEM CONTINUOUSLY OBSERVES
        ↓
ADAPTIVE LEARNING EXPERIENCE

The final website must feel like a REAL working AI prototype, not a static HTML demonstration.

Before finishing, verify that:

Application starts successfully

Frontend routes work

Backend endpoints work

Camera start/stop works

Processing start/stop works

Manual emotion selection works

Demo mode works

Cloud mode can be configured

Local mode can be configured

Auto fallback works

UI adaptation works

Failures are handled gracefully

If an implementation detail depends on the unknown structure of emotion_cnn.pkl or the unknown cloud API response, isolate that uncertainty behind a clean adapter/configuration layer instead of guessing.

Build the project now. --> -->