package com.gauthamkrishna.objectdetectyolo

import android.graphics.Color
import com.gauthamkrishna.objectdetectyolo.model.YoloV5ObjectDetector
import com.gauthamkrishna.objectdetectyolo.utils.CameraProcess
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.camera.view.PreviewView
import com.gauthamkrishna.objectdetectyolo.model.ImageAnalysis
import java.lang.Exception

class MainActivity : ComponentActivity() {
    private var yoloV5ObjectDetector: YoloV5ObjectDetector? = null
    private val cameraProcess = CameraProcess()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        window.statusBarColor = Color.TRANSPARENT
        val cameraPreviewMatch = findViewById<PreviewView>(R.id.camera_preview_match)
        cameraPreviewMatch.scaleType = PreviewView.ScaleType.FILL_START
        val boxLabelCanvas = findViewById<ImageView>(R.id.box_label_canvas)
        if (!cameraProcess.allPermissionsGranted(this)) {
            cameraProcess.requestPermissions(this)
        }
        val rotation = windowManager.defaultDisplay.rotation
        Log.i("image", "rotation: $rotation")
        cameraProcess.showCameraSupportSize(this@MainActivity)
        initModel()
        val fullScreenAnalyse = ImageAnalysis(
            cameraPreviewMatch,
            boxLabelCanvas,
            rotation,
            yoloV5ObjectDetector!!
        )
        cameraProcess.startCamera(this@MainActivity, fullScreenAnalyse, cameraPreviewMatch)
    }

    private fun initModel() {
        try {
            yoloV5ObjectDetector = YoloV5ObjectDetector()
            yoloV5ObjectDetector!!.addGPUDelegate()
            yoloV5ObjectDetector!!.initialModel(this)
            Log.i("model", "Success loading model" + yoloV5ObjectDetector!!.modelFile)
        } catch (e: Exception) {
            Log.e("image", "load model error: " + e.message + e.toString())
        }
    }
}