package com.gauthamkrishna.objectdetectyolo.model

import android.content.Context
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.common.FileUtil
import android.widget.Toast
import android.graphics.Bitmap
import com.gauthamkrishna.objectdetectyolo.utils.Recognition
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.DequantizeOp
import android.graphics.RectF
import org.tensorflow.lite.nnapi.NnApiDelegate
import android.os.Build
import android.util.Log
import android.util.Size
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.IOException
import java.nio.ByteBuffer
import java.util.*

class YoloV5ObjectDetector {

    val inputSize = Size(640, 640)
    private val outputSize = intArrayOf(1, 25200, 24)
    private val isInt8 = false
    private val detectThreshold = 0.25f
    private val iouThreshold = 0.45f
    private val iouClassDuplicatedThreshold = 0.7f
    private val labelFile = "labels_furniture.txt"
    var input5SINT8QuantParams = MetadataExtractor.QuantizationParams(0.003921568859368563f, 0)
    var output5SINT8QuantParams = MetadataExtractor.QuantizationParams(0.006305381190031767f, 5)
    val modelFile = "yolov5_50_furniture.tflite"
    private var tflite: Interpreter? = null
    private var associatedAxisLabels: List<String>? = null
    var options = Interpreter.Options()

    fun initialModel(activity: Context?) {
        // Initialise the model
        try {
            val tfliteModel: ByteBuffer = FileUtil.loadMappedFile(activity!!, modelFile)
            tflite = Interpreter(tfliteModel, options)
            Log.i("TFLITE", "Success reading model: $modelFile")
            associatedAxisLabels = FileUtil.loadLabels(activity, labelFile)
            Log.i("TFLITE", "Success reading label: $labelFile")
        } catch (e: IOException) {
            Log.e("TFLITE", "Error reading model or label: ", e)
            Toast.makeText(activity, "Error loading model: " + e.message, Toast.LENGTH_LONG).show()
        }
    }

    fun detect(bitmap: Bitmap?): ArrayList<Recognition?> {
        var yolov5sTfliteInput: TensorImage
        val imageProcessor: ImageProcessor
        if (isInt8) {
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .add(
                    QuantizeOp(
                        input5SINT8QuantParams.zeroPoint.toFloat(),
                        input5SINT8QuantParams.scale
                    )
                )
                .add(CastOp(DataType.UINT8))
                .build()
            yolov5sTfliteInput = TensorImage(DataType.UINT8)
        } else {
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()
            yolov5sTfliteInput = TensorImage(DataType.FLOAT32)
        }
        yolov5sTfliteInput.load(bitmap)
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput)
        var probabilityBuffer: TensorBuffer
        probabilityBuffer = if (isInt8) {
            TensorBuffer.createFixedSize(outputSize, DataType.UINT8)
        } else {
            TensorBuffer.createFixedSize(outputSize, DataType.FLOAT32)
        }
        if (null != tflite) {
            tflite!!.run(yolov5sTfliteInput.buffer, probabilityBuffer.buffer)
        }
        if (isInt8) {
            val tensorProcessor = TensorProcessor.Builder()
                .add(
                    DequantizeOp(
                        output5SINT8QuantParams.zeroPoint.toFloat(),
                        output5SINT8QuantParams.scale
                    )
                )
                .build()
            probabilityBuffer = tensorProcessor.process(probabilityBuffer)
        }
        val recognitionArray = probabilityBuffer.floatArray
        val allRecognitions = ArrayList<Recognition>()
        for (i in 0 until outputSize[1]) {
            val gridStride = i * outputSize[2]
            val x = recognitionArray[0 + gridStride] * inputSize.width
            val y = recognitionArray[1 + gridStride] * inputSize.height
            val w = recognitionArray[2 + gridStride] * inputSize.width
            val h = recognitionArray[3 + gridStride] * inputSize.height
            val xmin = 0.0.coerceAtLeast(x - w / 2.0).toInt()
            val ymin = 0.0.coerceAtLeast(y - h / 2.0).toInt()
            val xmax = inputSize.width.toDouble().coerceAtMost(x + w / 2.0).toInt()
            val ymax = inputSize.height.toDouble().coerceAtMost(y + h / 2.0).toInt()
            val confidence = recognitionArray[4 + gridStride]
            val classScores =
                Arrays.copyOfRange(recognitionArray, 5 + gridStride, outputSize[2] + gridStride)
            var labelId = 0
            var maxLabelScores = 0f
            for (j in classScores.indices) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j]
                    labelId = j
                }
            }
            val r = Recognition(
                labelId,
                "",
                maxLabelScores,
                confidence,
                RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
            )
            allRecognitions.add(
                r
            )
        }
        val nmsRecognitions = nms(allRecognitions)
        val nmsFilterBoxDuplicationRecognitions = nmsAllClass(nmsRecognitions)
        for (recognition in nmsFilterBoxDuplicationRecognitions) {
            val labelId = recognition!!.labelId
            val labelName = associatedAxisLabels!![labelId]
            recognition.labelName = labelName
        }
        return nmsFilterBoxDuplicationRecognitions
    }

    private fun nms(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition?> {
        val nmsRecognitions = ArrayList<Recognition?>()
        for (i in 0 until outputSize[2] - 5) {
            val pq = PriorityQueue(
                6300
            ) { l: Recognition?, r: Recognition? ->
                (r!!.confidence!!).compareTo(l!!.confidence!!)
            }
            for (j in allRecognitions.indices) {
                if (allRecognitions[j].labelId == i && allRecognitions[j].confidence!! > detectThreshold) {
                    pq.add(allRecognitions[j])
                }
            }
            while (pq.size > 0) {
                val a = arrayOfNulls<Recognition>(pq.size)
                val detections = pq.toArray(a)
                val max = detections[0]
                nmsRecognitions.add(max)
                pq.clear()
                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    if (boxIou(max!!.getLocation(), detection!!.getLocation()) < iouThreshold) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsRecognitions
    }

    protected fun nmsAllClass(allRecognitions: ArrayList<Recognition?>): ArrayList<Recognition?> {
        val nmsRecognitions = ArrayList<Recognition?>()
        val pq = PriorityQueue(
            100
        ) { l: Recognition?, r: Recognition? ->
            (r!!.confidence!!).compareTo(l!!.confidence!!)
        }
        for (j in allRecognitions.indices) {
            if (allRecognitions[j]!!.confidence!! > detectThreshold) {
                pq.add(allRecognitions[j])
            }
        }
        while (pq.size > 0) {
            val a = arrayOfNulls<Recognition>(pq.size)
            val detections = pq.toArray(a)
            val max = detections[0]
            nmsRecognitions.add(max)
            pq.clear()
            for (k in 1 until detections.size) {
                val detection = detections[k]
                if (boxIou(
                        max!!.getLocation(),
                        detection!!.getLocation()
                    ) < iouClassDuplicatedThreshold
                ) {
                    pq.add(detection)
                }
            }
        }
        return nmsRecognitions
    }

    protected fun boxIou(a: RectF, b: RectF): Float {
        val intersection = boxIntersection(a, b)
        val union = boxUnion(a, b)
        return if (union <= 0) 1f else intersection / union
    }

    protected fun boxIntersection(a: RectF, b: RectF): Float {
        val maxLeft = if (a.left > b.left) a.left else b.left
        val maxTop = if (a.top > b.top) a.top else b.top
        val minRight = if (a.right < b.right) a.right else b.right
        val minBottom = if (a.bottom < b.bottom) a.bottom else b.bottom
        val w = minRight - maxLeft
        val h = minBottom - maxTop
        return if (w < 0 || h < 0) 0f else w * h
    }

    protected fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    fun addNNApiDelegate() {
        var nnApiDelegate: NnApiDelegate? = null
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            nnApiDelegate = NnApiDelegate()
            options.addDelegate(nnApiDelegate)
            Log.i("TFLITE", "using nnapi delegate.")
        }
    }

    fun addGPUDelegate() {
        val compatibilityList = CompatibilityList()
        if (compatibilityList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = compatibilityList.bestOptionsForThisDevice
            val gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate)
            Log.i("TFLITE", "using gpu delegate.")
        } else {
            addThread(4)
        }
    }

    fun addThread(thread: Int) {
        options.numThreads = thread
    }
}