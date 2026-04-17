import re


class SafetyFilter:
    def __init__(self):
        self.diagnosis_patterns = [
            r"\bdo i have\b",
            r"\bam i having\b",
            r"\bdo i suffer from\b",
            r"\bdo i need\b.*diagnosis",
            r"\bwhat disease\b.*i have",
            r"\bam i sick\b.*\bwith\b",
            r"\bcheck\b.*\bsymptoms?\b",
            r"\bdiagnose\b",
        ]
        self.prescription_patterns = [
            r"\bprescribe\b",
            r"\bprescription\b",
            r"\bgive me\b.*\bmedicine\b",
            r"\bshould i take\b.*\bdosage\b",
            r"\bhow much\b.*\bmedication\b",
            r"\brecommend\b.*\bmedicine\b",
            r"\bwhich\b.*\bdrug\b",
            r"\bantibiotic\b",
        ]
        self.dangerous_patterns = [
            r"\bsuicide\b",
            r"\bself.?harm\b",
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\btake\b.*\boverdose\b",
        ]
        self.emergency_keywords = [
            "chest pain", "heart attack", "stroke", "cannot breathe",
            "severe bleeding", "unconscious", "seizure",
        ]

    def is_diagnosis_request(self, query):
        query_lower = query.lower()
        for pattern in self.diagnosis_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_prescription_request(self, query):
        query_lower = query.lower()
        for pattern in self.prescription_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_dangerous_request(self, query):
        query_lower = query.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_emergency(self, query):
        query_lower = query.lower()
        for keyword in self.emergency_keywords:
            if keyword in query_lower:
                return True
        return False

    def filter_query(self, query):
        if self.is_emergency(query):
            return {
                "allowed": False,
                "response": "This sounds like a medical emergency. Please call emergency services (911) or go to your nearest emergency room immediately. I'm just an AI and cannot help with emergencies."
            }
        
        if self.is_dangerous_request(query):
            return {
                "allowed": False,
                "response": "I'm not able to help with this type of request. Please reach out to a crisis helpline or healthcare professional immediately. If you're in crisis, call 988 (Suicide & Crisis Lifeline)."
            }
        
        if self.is_diagnosis_request(query):
            return {
                "allowed": False,
                "response": "I'm not a doctor and cannot provide medical diagnoses. Please consult a healthcare professional for any health concerns. They can properly evaluate your symptoms and provide appropriate care."
            }
        
        if self.is_prescription_request(query):
            return {
                "allowed": False,
                "response": "I'm not able to prescribe medication or recommend specific dosages. Please consult a doctor or pharmacist for medication-related questions. They can provide personalized advice based on your medical history."
            }
        
        return {"allowed": True, "response": None}