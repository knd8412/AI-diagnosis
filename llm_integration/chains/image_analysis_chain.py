from llm_integration.service.image_processing_client import analyze_xray_image, extract_pathology_symptoms
from llm_integration.chains.rag_chains import DiagnosisRAG
from llm_integration.chains.explanation_chain import DiagnosisExplainer


class ImageAnalysisChain:

    def __init__(self):
        try:
            self.rag = DiagnosisRAG()
            self.explainer = DiagnosisExplainer()
        except Exception as e:
            raise RuntimeError(f"Failed to initialise ImageAnalysisChain: {e}")

    def analyze_with_xray(
        self,
        image_path: str,
        patient_id: str,
        patient_text: str = "",
        chat_history: str = ""
    ) -> dict:
        """
        Complete workflow: analyze X-ray → diagnose → return results.

        Args:
            image_path:   Path to uploaded X-ray
            patient_id:   ID of the current patient
            patient_text: Optional patient description
            chat_history: Previous conversation

        Returns:
            dict with keys: type, xray_findings, pathology_description,
                            diagnosis, retrieved_context, scan_id
        """
        try:
            xray_result = analyze_xray_image(image_path)
        except RuntimeError as e:
            return {
                "type": "error",
                "message": str(e)
            }

        pathology_description = extract_pathology_symptoms(xray_result)

        if not xray_result.get("confident_results"):
            return {
                "type": "clarification_needed",
                "message": (
                    "No significant findings detected in the X-ray. "
                    "Please provide more information about your symptoms "
                    "or upload a clearer image."
                ),
                "xray_analysis": xray_result
            }

        combined_query = pathology_description
        if patient_text:
            combined_query += f"\n\nPatient also reports: {patient_text}"

        diagnosis_result = self.rag.diagnose(
            patient_id=patient_id,
            patient_query=combined_query,
            chat_history=chat_history
        )

        return {
            "type": "diagnosis_with_xray",
            "xray_findings": xray_result["confident_results"],
            "pathology_description": pathology_description,
            "diagnosis": diagnosis_result["diagnosis"],
            "retrieved_context": diagnosis_result["retrieved_context"],
            "scan_id": xray_result.get("result_id")
        }

    def explain_xray_findings(
        self,
        image_path: str,
        specific_pathology: str = None
    ) -> dict:
        """
        Explain specific pathology from X-ray.

        Args:
            image_path:         Path to X-ray
            specific_pathology: Specific condition to explain (optional)

        Returns:
            dict with keys: type, explanation (or message on failure)
        """
        try:
            xray_result = analyze_xray_image(image_path)
        except RuntimeError as e:
            return {
                "type": "error",
                "message": str(e)
            }

        confident = xray_result.get("confident_results", {})

        if not confident:
            return {
                "type": "clarification_needed",
                "message": "No significant pathologies detected to explain."
            }

        if specific_pathology and specific_pathology in confident:
            condition = specific_pathology
        else:
            condition = max(confident, key=confident.get)

        pathology_description = extract_pathology_symptoms(xray_result)
        context = self.rag.retrieve_context(condition)

        explanation = self.explainer.explain(
            condition=condition,
            patient_symptoms=pathology_description,
            context=context
        )

        return {
            "type": "explanation",
            "condition": condition,
            "explanation": explanation
        }