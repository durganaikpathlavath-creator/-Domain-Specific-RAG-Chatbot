"""create_sample_documents.py
Generates realistic multi-page PDF documents for testing the RAG chatbot:
- documents/Company_Policy.pdf (6 pages)
- documents/Employee_Handbook.pdf (11 pages)
"""

import os
from pathlib import Path


def generate_pdf(filepath: str, pages_content: list[str]) -> None:
    """Generate a clean, standard PDF 1.4 file with Helvetica font and page streams."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    all_objs = {}
    all_objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    all_objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    next_id = 4
    page_ids = []

    for text in pages_content:
        stream_lines = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
        for line in text.strip().split("\n"):
            escaped = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            stream_lines.append(f"({escaped}) Tj T*")
        stream_lines.append("ET")

        content_str = "\n".join(stream_lines)
        content_bytes = content_str.encode("latin-1", errors="replace")

        c_id = next_id
        next_id += 1
        all_objs[c_id] = f"<< /Length {len(content_bytes)} >>\nstream\n{content_str}\nendstream"

        p_id = next_id
        next_id += 1
        page_ids.append(p_id)
        all_objs[p_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )

    kids_str = " ".join([f"{pid} 0 R" for pid in page_ids])
    all_objs[2] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>"

    with open(filepath, "wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = {}
        for obj_id in sorted(all_objs.keys()):
            offsets[obj_id] = f.tell()
            f.write(f"{obj_id} 0 obj\n{all_objs[obj_id]}\nendobj\n".encode("latin-1"))
        xref_pos = f.tell()
        f.write(f"xref\n0 {len(all_objs) + 1}\n".encode("latin-1"))
        f.write(b"0000000000 65535 f \n")
        for obj_id in sorted(all_objs.keys()):
            f.write(f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1"))
        f.write(
            f"trailer\n<< /Size {len(all_objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
                "latin-1"
            )
        )


def main():
    docs_dir = Path(__file__).resolve().parent / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Company_Policy.pdf (6 pages)
    policy_pages = [
        # Page 1
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 1: General Policy & Equal Opportunity

1.1 Purpose & Scope
This document sets forth general company policies governing all employees,
contractors, and officers across all corporate branches. Acme Enterprises is
committed to fostering an ethical, transparent, and collaborative environment.

1.2 Equal Employment Opportunity
Acme provides equal employment opportunities to all employees and applicants
regardless of race, color, religion, sex, national origin, age, disability,
genetic information, or sexual orientation. Hiring and advancement decisions
are made solely on merit, qualifications, and business operational needs.""",

        # Page 2
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 2: Working Hours & Overtime Guidelines

2.1 Standard Working Hours
Standard full-time business hours are 9:00 AM to 5:00 PM local time, Monday
through Friday. All teams observe Core Collaboration Hours between 10:00 AM
and 4:00 PM, during which cross-functional meetings and team syncs take place.

2.2 Overtime Compensation
Non-exempt employees who work exceeding 40 hours during an official workweek
receive overtime pay calculated at 1.5 times their standard hourly rate.
All non-exempt overtime requires prior approval in writing from the department head.""",

        # Page 3
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 3: Remote Work & Technology Allowances

3.1 Hybrid & Remote Work Eligibility
Eligible team members may work remotely up to two days per week with departmental
approval. Full-time remote arrangements require executive VP approval.

3.2 Technology & Home Office Allowance
Full-time remote employees receive a one-time home office equipment stipend of
$500 upon joining, plus a monthly internet reimbursement subsidy of $50.""",

        # Page 4
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 4: Travel & Business Expense Reimbursement

4.1 Travel Authorization
All domestic and international corporate travel must be pre-approved via the
internal travel portal at least 14 days prior to departure.

4.2 Expense Claims & Per Diem
Employees receive a per diem rate of $75 for meals while traveling on approved
business. Itemized receipts are required for all individual expenses exceeding $25.
Expense reports must be submitted within 30 days of trip conclusion.""",

        # Page 5
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 5: Information Security & Data Protection

5.1 Password & Access Protocols
All employees must utilize multi-factor authentication (MFA) on corporate accounts.
Passwords must be at least 12 characters in length and rotated every 90 days.

5.2 Confidentiality & Whistleblower Protection
Proprietary corporate data must never be transmitted via unauthorized third-party
messaging applications. Employees may report suspected ethical breaches confidentially
via the dedicated hotline without fear of retaliation.""",

        # Page 6
        """ACME ENTERPRISES - CORPORATE POLICY MANUAL
Section 6: Comprehensive Leave Policy

6.1 Annual Leave (Paid Time Off)
Full-time personnel receive 20 days of paid annual vacation leave per calendar year,
accrued on a monthly pro-rata basis. A maximum of 5 unused days may roll over to the next year.

6.2 Sick Leave
Employees are entitled to 10 paid sick leave days per calendar year. For sick leave
exceeding 3 consecutive business days, a certified medical practitioner note is mandatory.

6.3 Parental Leave
Primary caregivers are entitled to 16 weeks of fully paid parental leave following
the birth, adoption, or foster placement of a child. Secondary caregivers receive 4 weeks paid leave.

6.4 Bereavement Leave
Employees are provided up to 5 consecutive paid business days for immediate family members
and up to 3 days for extended family members."""
    ]

    policy_pdf_path = docs_dir / "Company_Policy.pdf"
    generate_pdf(str(policy_pdf_path), policy_pages)
    print(f"Created {policy_pdf_path} (6 pages)")

    # 2. Employee_Handbook.pdf (11 pages)
    handbook_pages = [
        # Page 1
        """ACME EMPLOYEE HANDBOOK
Chapter 1: Welcome to Acme Enterprises

Welcome to Acme Enterprises! Our mission is to deliver innovative technological
solutions with absolute integrity and customer-centric excellence.
This handbook introduces key operational practices, workplace resources, and
cultural principles that make our company a premier place to build a rewarding career.""",

        # Page 2
        """ACME EMPLOYEE HANDBOOK
Chapter 2: Probationary Period & Onboarding

New full-time hires undertake a 90-day introductory probationary period.
During this window, managers provide structured 30-day and 60-day performance check-ins.
A formal milestone review is conducted at day 90 to confirm ongoing regular employment status.""",

        # Page 3
        """ACME EMPLOYEE HANDBOOK
Chapter 3: Payroll Schedule & Compensation

Salaries are distributed bi-weekly on alternate Fridays via electronic direct deposit.
If a scheduled payday falls on an official national banking holiday, funds are
transferred on the immediately preceding business day.""",

        # Page 4
        """ACME EMPLOYEE HANDBOOK
Chapter 4: Health, Dental & Vision Insurance Benefits

Comprehensive medical, dental, and vision insurance coverage commences on the first
day of the month following the hire date. Acme covers 80% of individual medical premiums
and 60% of family dependent premiums. Open enrollment occurs annually in November.""",

        # Page 5
        """ACME EMPLOYEE HANDBOOK
Chapter 5: Professional Development & Tuition Subsidy

Acme encourages continuous learning. Regular employees who complete one year of service
are eligible for up to $2,500 annually in tuition reimbursement for accredited degree
programs, technical certifications, or approved industry conferences.""",

        # Page 6
        """ACME EMPLOYEE HANDBOOK
Chapter 6: Health, Safety & Emergency Protocols

Safety is our collective responsibility. All office facilities maintain automated external
defibrillators (AED) and first aid kits on each floor. In the event of a fire alarm or
evacuation notice, exit calmly via stairwells to designated outdoor assembly points.""",

        # Page 7
        """ACME EMPLOYEE HANDBOOK
Chapter 7: Professional Code of Conduct & Anti-Harassment

Acme maintains zero tolerance for harassment, discrimination, or abusive conduct of any kind.
Professional decorum, mutual respect, and courteous communication are required at all times
across in-person offices, virtual meetings, and electronic communications.""",

        # Page 8
        """ACME EMPLOYEE HANDBOOK
Chapter 8: Social Media & Public Communications

Employees speaking on personal social media channels must clarify that opinions expressed
are strictly their own and not representative of Acme Enterprises. Only official corporate
spokespersons designated by Executive Management may speak to press or news media.""",

        # Page 9
        """ACME EMPLOYEE HANDBOOK
Chapter 9: Intellectual Property & Inventions

All inventions, software code, algorithms, designs, and documentation developed during working
hours or utilizing Acme equipment remain the exclusive intellectual property of Acme Enterprises.
Employees execute an Inventions Assignment Agreement upon hire.""",

        # Page 10
        """ACME EMPLOYEE HANDBOOK
Chapter 10: Performance Management & Reviews

Performance evaluations are conducted bi-annually: a mid-year check-in during June and
an annual review in December. Performance ratings influence annual merit salary adjustments,
promotional eligibility, and annual performance incentive bonuses.""",

        # Page 11
        """ACME EMPLOYEE HANDBOOK
Chapter 11: Attendance Tracking and Calculation

11.1 Attendance Calculation Methodology
Attendance is calculated based on automated electronic badge swipes at office entry turnstiles
and active single sign-on (SSO) login sessions on corporate laptops. Full-time personnel are
expected to complete 40 accountable working hours per week.

11.2 Grace Periods and Late Arrival
A standard 15-minute grace period is provided for morning check-ins. If an employee arrives
more than 30 minutes after standard shift start without prior supervisor notice, it is logged
as an unscheduled late arrival. Three unscheduled late arrivals in a calendar month trigger an informal review."""
    ]

    handbook_pdf_path = docs_dir / "Employee_Handbook.pdf"
    generate_pdf(str(handbook_pdf_path), handbook_pages)
    print(f"Created {handbook_pdf_path} (11 pages)")


if __name__ == "__main__":
    main()
