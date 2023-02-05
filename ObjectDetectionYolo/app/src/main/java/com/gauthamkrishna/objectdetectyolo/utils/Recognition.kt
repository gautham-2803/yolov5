package com.gauthamkrishna.objectdetectyolo.utils

import android.graphics.RectF

class Recognition(
    var labelId: Int,
    var labelName: String?,
    var labelScore: Float,
    var confidence: Float?,
    private var location: RectF?
) {

    fun getLocation(): RectF {
        return RectF(location)
    }

    fun setLocation(location: RectF?) {
        this.location = location
    }

    override fun toString(): String {
        var resultString = ""
        resultString += "$labelId "
        if (labelName != null) {
            resultString += "$labelName "
        }
        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence!! * 100.0f)
        }
        if (location != null) {
            resultString += location.toString() + " "
        }
        return resultString.trim { it <= ' ' }
    }
}