package io.github.cyberneticmedic.transferner.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** The JSON files of a model bundle, as written by src/transferner/export/bundle.py. */
val BundleJson = Json { ignoreUnknownKeys = true }

@Serializable
data class Latest(val version: String, val manifest: String)

@Serializable
data class Manifest(val version: String, val created: String = "", val files: List<ManifestFile>)

@Serializable
data class ManifestFile(val path: String, val bytes: Long, val sha256: String, val group: String? = null) {
    /** Files of one species-model variant carry group "species:<id>"; the rest are needed by every phone. */
    fun wantedFor(variant: String) = group == null || group == "species:$variant"
}

@Serializable
data class TensorInfo(
    val name: String,
    val dtype: String,
    val shape: List<Int>,
    val scale: Float = 0f,
    @SerialName("zero_point") val zeroPoint: Int = 0,
) {
    val isInt8 get() = dtype == "int8"
}

@Serializable
data class DetectorOutputs(
    @SerialName("image_logits") val imageLogits: TensorInfo,
    val objectness: TensorInfo,
    val boxes: TensorInfo,
)

@Serializable
data class BoxCoder(val weights: List<Float>, val clip: Float)

@Serializable
data class Postprocess(
    @SerialName("objectness_topk") val objectnessTopK: Int,
    @SerialName("nms_iou") val nmsIou: Float,
    @SerialName("max_boxes") val maxBoxes: Int,
)

@Serializable
data class DetectorMeta(
    val model: String,
    val signature: String,
    val size: Int,
    val mean: List<Float>,
    val std: List<Float>,
    val input: TensorInfo,
    val outputs: DetectorOutputs,
    @SerialName("box_coder") val boxCoder: BoxCoder,
    val anchors: List<List<Float>>,
    val postprocess: Postprocess,
    val genera: List<String>,
    val families: List<String>,
    @SerialName("family_of") val familyOf: List<Int>,
)

@Serializable
data class SpeciesPreprocess(val mean: List<Float>, val std: List<Float>)

/** One species-model variant (D-032/D-035): "gpu" W8A16, "cpu" dynamic INT8, "cpu-small" INT4 + dynamic INT8. */
@Serializable
data class SpeciesModelMeta(
    val id: String,
    val label: String,
    val model: String,
    val variant: String,
    val accelerator: String,                                // GPU | CPU: what the variant is built for
    @SerialName("min_ram_gb") val minRamGb: Float = 0f,
    val bytes: Long = 0,
    val signature: String,
    val input: TensorInfo,
    val output: TensorInfo,
    val preprocess: SpeciesPreprocess,
)

@Serializable
data class SpeciesModels(val variants: List<SpeciesModelMeta>) {
    operator fun get(id: String) = variants.firstOrNull { it.id == id }
}

@Serializable
data class HeadMeta(
    val file: String,
    val n: Int,
    val dim: Int,
    val scale: Float,
    val temperature: Float,
    val bias: List<Float>,
)

@Serializable
data class Trait(
    val part: String,
    val category: String,                                   // toxic | edible | conditional
    @SerialName("contact_toxic") val contactToxic: Int = 0,
    @SerialName("ingestion_toxic") val ingestionToxic: Int = 0,
    val severity: String = "",
    val conditions: String = "",
    @SerialName("toxic_principle") val toxicPrinciple: String = "",
    val symptoms: String = "",
    @SerialName("genus_level") val genusLevel: Boolean = false,
    val remarks: List<String> = emptyList(),
    val source: String = "",
    val url: String = "",
    val accessed: String = "",
)

@Serializable
data class Thumb(val file: String, val by: String, val license: String, val ref: String = "")

@Serializable
data class Species(
    val i: Int,
    val name: String,
    val common: String = "",
    val genus: String,
    val family: String,
    @SerialName("lookalike_groups") val lookalikeGroups: List<String> = emptyList(),
    val traits: List<Trait> = emptyList(),
    val thumb: Thumb? = null,
) {
    /** Common name when the catalog has one, else the scientific name. */
    val displayName get() = common.ifBlank { name }
}

@Serializable
data class Catalog(val species: List<Species>, val region: List<String> = emptyList())

/** focus.json (eval/focus_prior.py, D-037): genera that can take each growth form, and the soft-prior strength. */
@Serializable
data class FocusCategory(val id: String, val label: String, val genera: List<Int>)

@Serializable
data class FocusMeta(val eps: Float, val categories: List<FocusCategory>, val source: String = "") {
    /** Per-genus logit bias for [id]: 0 inside the category (or without habit data), ln(eps) outside. */
    fun bias(id: String, nGenera: Int): FloatArray? {
        val c = categories.firstOrNull { it.id == id } ?: return null
        val out = if (eps <= 0f) -1e9f else kotlin.math.ln(eps)
        val b = FloatArray(nGenera) { out }
        for (g in c.genera) if (g in 0 until nGenera) b[g] = 0f
        return b
    }
}

/** app.json on the bundle server (scripts/publish_apk.py, D-042): the latest sideloaded APK. */
@Serializable
data class AppRelease(
    val versionCode: Long,
    val versionName: String,
    val apk: String,
    val bytes: Long,
    val sha256: String,
    val notes: String = "",
) {
    fun newerThan(installed: Long) = versionCode > installed
}
