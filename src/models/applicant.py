from typing import Dict
from dataclasses import dataclass


@dataclass
class Applicant:
    """Applicant data model"""
    email: str
    password: str
    passport_number: str
    first_name: str
    last_name: str
    date_of_birth: str
    nationality: str
    phone: str
    visa_type: str
    center: str
    travel_date: str
    duration: str
    purpose: str
    location: str
    visa_type_option: str
    visa_sub_type_option: str
    appointment_for: str
    members_count: str
    category: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Applicant':
        """Create Applicant from dictionary"""
        return cls(
            email=data.get('email', ''),
            password=data.get('password', ''),
            passport_number=data.get('passport_number', ''),
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            date_of_birth=data.get('date_of_birth', ''),
            nationality=data.get('nationality', ''),
            phone=data.get('phone', ''),
            visa_type=data.get('visa_type', ''),
            center=data.get('center', ''),
            travel_date=data.get('travel_date', ''),
            duration=data.get('duration', ''),
            purpose=data.get('purpose', ''),
            location=data.get('Location', '') or data.get('location', ''),
            visa_type_option=data.get('Visa Type', '') or data.get('visa_type_option', ''),
            visa_sub_type_option=data.get('Visa Sub Type', '') or data.get('visa_sub_type_option', ''),
            appointment_for=data.get('Appointment For', '') or data.get('appointment_for', ''),
            members_count=data.get('Number Of Members', '') or data.get('members_count', ''),
            category=data.get('Category', '') or data.get('Categoty', '') or data.get('category', '')
        )
    
    def get_full_name(self) -> str:
        """Get full name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'email': self.email,
            'password': self.password,
            'passport_number': self.passport_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'date_of_birth': self.date_of_birth,
            'nationality': self.nationality,
            'phone': self.phone,
            'visa_type': self.visa_type,
            'center': self.center,
            'travel_date': self.travel_date,
            'duration': self.duration,
            'purpose': self.purpose,
            'location': self.location,
            'visa_type_option': self.visa_type_option,
            'visa_sub_type_option': self.visa_sub_type_option,
            'appointment_for': self.appointment_for,
            'members_count': self.members_count,
            'category': self.category
        }
