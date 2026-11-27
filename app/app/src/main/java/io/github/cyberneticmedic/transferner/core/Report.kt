package io.github.cyberneticmedic.transferner.core

import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject

/**
 * The device report uploaded with the log (D-036): flat key/value pairs as one JSON object. Numbers and
 * booleans stay JSON numbers and booleans, anything else becomes its string, and null values are left out.
 *
 * 0.6-0.7.1 built this inside buildJsonObject with a local helper named put(k, Any?). Kotlin binds the helper's
 * own put(k, JsonPrimitive(..)) calls to the helper itself (a local function outranks the builder's member), so
 * it recursed, re-quoting the string at every level until the app ran out of memory and closed. That happened on
 * every "Send", before anything was sent. The builder is now only called here, through its member put.
 */
object Report {
    fun json(fields: List<Pair<String, Any?>>): String = buildJsonObject {
        for ((k, v) in fields) {
            when (v) {
                null -> {}
                is Number -> put(k, JsonPrimitive(v))
                is Boolean -> put(k, JsonPrimitive(v))
                else -> put(k, JsonPrimitive(v.toString()))
            }
        }
    }.toString()
}
