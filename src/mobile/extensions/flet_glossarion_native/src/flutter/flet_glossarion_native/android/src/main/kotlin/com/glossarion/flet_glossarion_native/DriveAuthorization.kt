package com.glossarion.flet_glossarion_native

import android.app.Activity
import android.content.Context
import android.content.Intent
import com.google.android.gms.auth.api.identity.AuthorizationRequest
import com.google.android.gms.auth.api.identity.AuthorizationResult
import com.google.android.gms.auth.api.identity.ClearTokenRequest
import com.google.android.gms.auth.api.identity.Identity
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.common.api.Scope
import io.flutter.plugin.common.MethodChannel

/** Play services identifies Glossarion by its package and signing certificate. */
internal class DriveAuthorization {
    private val requestCode = 6291
    private var pending: MethodChannel.Result? = null
    private val request = AuthorizationRequest.builder()
        .setRequestedScopes(listOf(Scope("https://www.googleapis.com/auth/drive.file")))
        .build()

    fun authorize(context: Context, activity: Activity?, interactive: Boolean, rejected: String?,
                  result: MethodChannel.Result) {
        if (pending != null) {
            result.success(mapOf("error" to "busy"))
            return
        }
        val client = Identity.getAuthorizationClient(activity ?: context)
        pending = result
        fun requestToken() {
            client.authorize(request).addOnSuccessListener { response ->
                if (!response.hasResolution()) {
                    finish(response)
                } else if (!interactive || activity == null) {
                    finish(error = "sign_in")
                } else {
                    try {
                        activity.startIntentSenderForResult(response.pendingIntent!!.intentSender, requestCode,
                            null, 0, 0, 0)
                    } catch (_: Exception) {
                        finish(error = "failed")
                    }
                }
            }.addOnFailureListener { finish(error = errorCode(it)) }
        }
        if (rejected.isNullOrEmpty()) requestToken()
        else client.clearToken(ClearTokenRequest.builder().setToken(rejected).build())
            .addOnSuccessListener { requestToken() }
            .addOnFailureListener { finish(error = errorCode(it)) }
    }

    fun onActivityResult(context: Context?, code: Int, data: Intent?): Boolean {
        if (code != requestCode || pending == null) return false
        if (context == null || data == null) {
            finish(error = "cancelled")
        } else {
            try {
                finish(Identity.getAuthorizationClient(context).getAuthorizationResultFromIntent(data))
            } catch (error: Exception) {
                finish(error = errorCode(error))
            }
        }
        return true
    }

    fun finish(response: AuthorizationResult? = null, error: String = "sign_in") {
        val result = pending
        pending = null
        val token = response?.accessToken
        result?.success(if (token.isNullOrEmpty()) mapOf("error" to error) else mapOf("token" to token))
    }

    private fun errorCode(error: Exception): String =
        (error as? ApiException)?.statusCode?.let { if (it == 16 || it == 12501) "cancelled" else "google_$it" }
            ?: "failed"
}
