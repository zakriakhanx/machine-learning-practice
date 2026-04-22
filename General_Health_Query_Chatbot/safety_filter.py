"""
Safety Filter Module for the Health Query Chatbot.

This module provides content filtering to ensure the chatbot does not provide
unsafe or inappropriate health-related responses. It implements multiple layers
of protection:

1. Emergency Detection - Identifies medical emergencies requiring immediate help
2. Dangerous Request Detection - Identifies self-harm/crisis situations  
3. Diagnosis Prevention -Blocks requests for medical diagnoses
4. Prescription Prevention - Blocks requests for medication/prescriptions

Each category returns a safe response directing users to appropriate
professional help rather than attempting to answer potentially harmful queries.
"""

import re


class SafetyFilter:
    """
    Safety filter class that screens user queries for potentially harmful
    health requests and redirects to professional help when needed.
    """

    def __init__(self):
        """
        Initialize the SafetyFilter with regex patterns and keywords.
        
        Patterns are organized into categories:
        - diagnosis_patterns: Phrases requesting medical diagnoses
        - prescription_patterns: Requests for medications or dosages
        - dangerous_patterns: Crisis/self-harm related language
        - emergency_keywords: Medical emergency indicators
        """
        # Patterns that indicate a diagnosis request
        # Matches phrases like "do I have", "am I having", "diagnose me", etc.
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
        
        # Patterns that indicate a prescription request
        # Matches phrases like "prescribe", "give me medicine", "dosage", etc.
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
        
        # Patterns that indicate dangerous/crisis situations
        # Matches self-harm, suicide, or overdose language
        self.dangerous_patterns = [
            r"\bsuicide\b",
            r"\bself.?harm\b",
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\btake\b.*\boverdose\b",
        ]
        
        # Emergency keywords that require immediate professional attention
        # These are simple substring matches (not regex) for faster checking
        self.emergency_keywords = [
            "chest pain", "heart attack", "stroke", "cannot breathe",
            "severe bleeding", "unconscious", "seizure",
        ]

    def is_diagnosis_request(self, query):
        """
        Check if the query is requesting a medical diagnosis.
        
        Args:
            query: The user's input string
            
        Returns:
            bool: True if the query appears to request a diagnosis
        """
        query_lower = query.lower()
        for pattern in self.diagnosis_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_prescription_request(self, query):
        """
        Check if the query is requesting medication or prescription info.
        
        Args:
            query: The user's input string
            
        Returns:
            bool: True if the query requests prescription/medication info
        """
        query_lower = query.lower()
        for pattern in self.prescription_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_dangerous_request(self, query):
        """
        Check if the query indicates a crisis or self-harm situation.
        
        Args:
            query: The user's input string
            
        Returns:
            bool: True if the query suggests dangerous/self-harm intent
        """
        query_lower = query.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def is_emergency(self, query):
        """
        Check if the query describes a medical emergency.
        
        Args:
            query: The user's input string
            
        Returns:
            bool: True if the query mentions emergency medical symptoms
        """
        query_lower = query.lower()
        for keyword in self.emergency_keywords:
            if keyword in query_lower:
                return True
        return False

    def filter_query(self, query):
        """
        Main filtering method that checks query against all safety categories.
        
        This method runs through safety checks in order of priority:
        1. Emergency -highest priority, direct to emergency services
        2. Dangerous - crisis situations, direct to crisis hotlines
        3. Diagnosis - not a doctor, redirect to professionals
        4. Prescription - cannot prescribe, redirect to doctors
        
        Args:
            query: The user's input string
            
        Returns:
            dict: Contains 'allowed' (bool) and 'response' (str or None)
                  - If allowed=False, 'response' contains the safety message
                  - If allowed=True, 'response' is None
        """
        # Priority 1: Medical emergencies
        # These require immediate professional intervention
        if self.is_emergency(query):
            return {
                "allowed": False,
                "response": "This sounds like a medical emergency. Please call emergency services (911) or go to your nearest emergency room immediately. I'm just an AI and cannot help with emergencies."
            }
        
        # Priority 2: Dangerous/crisis situations
        # These require mental health or crisis support
        if self.is_dangerous_request(query):
            return {
                "allowed": False,
                "response": "I'm not able to help with this type of request. Please reach out to a crisis helpline or healthcare professional immediately. If you're in crisis, call 988 (Suicide & Crisis Lifeline)."
            }
        
        # Priority 3: Diagnosis requests
        # Only qualified healthcare professionals can diagnose
        if self.is_diagnosis_request(query):
            return {
                "allowed": False,
                "response": "I'm not a doctor and cannot provide medical diagnoses. Please consult a healthcare professional for any health concerns. They can properly evaluate your symptoms and provide appropriate care."
            }
        
        # Priority 4: Prescription requests
        # Only doctors can prescribe medication
        if self.is_prescription_request(query):
            return {
                "allowed": False,
                "response": "I'm not able to prescribe medication or recommend specific dosages. Please consult a doctor or pharmacist for medication-related questions. They can provide personalized advice based on your medical history."
            }
        
        # Query passes all safety checks
        return {"allowed": True, "response": None}