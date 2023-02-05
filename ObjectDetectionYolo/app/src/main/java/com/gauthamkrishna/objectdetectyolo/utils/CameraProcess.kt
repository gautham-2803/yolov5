package com.gauthamkrishna.objectdetectyolo.utils

import com.google.common.util.concurrent.ListenableFuture
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import android.app.Activity
import android.content.Context
import androidx.core.app.ActivityCompat
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.PreviewView
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.lifecycle.LifecycleOwner
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraCharacteristics
import android.graphics.SurfaceTexture
import android.util.Log
import androidx.camera.core.Preview
import java.lang.Exception
import java.util.concurrent.ExecutionException

class CameraProcess {
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null
    private val REQUEST_CODE_PERMISSIONS = 1001
    private val REQUIRED_PERMISSIONS = arrayOf(
        "android.permission.CAMERA",
        "android.permission.WRITE_EXTERNAL_STORAGE"
    )

    fun allPermissionsGranted(context: Context?): Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(
                    context!!,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

    fun requestPermissions(activity: Activity?) {
        ActivityCompat.requestPermissions(
            activity!!,
            REQUIRED_PERMISSIONS,
            REQUEST_CODE_PERMISSIONS
        )
    }

    fun startCamera(
        context: Context?,
        analyzer: ImageAnalysis.Analyzer?,
        previewView: PreviewView
    ) {
        cameraProviderFuture = ProcessCameraProvider.getInstance(context!!)
        cameraProviderFuture!!.addListener({
            try {
                val cameraProvider = cameraProviderFuture!!.get()
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer!!)
                val previewBuilder = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
                previewBuilder.setSurfaceProvider(previewView.createSurfaceProvider())
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    (context as LifecycleOwner?)!!,
                    cameraSelector,
                    imageAnalysis,
                    previewBuilder
                )
            } catch (e: ExecutionException) {
                e.printStackTrace()
            } catch (e: InterruptedException) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun showCameraSupportSize(activity: Activity) {
        val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (id in manager.cameraIdList) {
                val cc = manager.getCameraCharacteristics(id!!)
                if (cc.get(CameraCharacteristics.LENS_FACING) == 1) {
                    val previewSizes = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        ?.getOutputSizes(SurfaceTexture::class.java)
                    previewSizes?.let {
                        for (s in previewSizes) {
                            Log.i("camera", s.height.toString() + "/" + s.width)
                        }
                    }
                    break
                }
            }
        } catch (e: Exception) {
            Log.e("image", "can not open camera", e)
        }
    }
}