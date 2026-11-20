##create 60 test questions across six policy categories##
import json
from pathlib import Path

DATASET = {
    "flight_policy": [
        (
            "Can a standard employee book business class?",
            "Standard employees must book economy class. Business class is not permitted under the sample policy."
        ),
        (
            "Is business class allowed for executives?",
            "Yes. Executive employees are permitted to book business class."
        ),
        (
            "What cabin class should a contractor book?",
            "Contractors must book economy class."
        ),
        (
            "Can a contractor book business class?",
            "No. Contractors must book economy class."
        ),
        (
            "How early should I book my flight?",
            "Flights should normally be booked at least 14 calendar days before departure."
        ),
        (
            "I booked my flight 5 days before departure. What happens?",
            "Bookings within 14 days require business justification and manager review."
        ),
        (
            "Can an executive book a flight 3 days before travel?",
            "The 14-day advance booking rule applies. A booking within 14 days requires business justification and manager review."
        ),
        (
            "Can I get an exception for business class?",
            "An international business-class exception requires written executive authorization. The agent cannot approve it."
        ),
        (
            "What is the flight policy for regular employees?",
            "Regular full-time employees are classified as Standard and must book economy class."
        ),
        (
            "Is a last-minute flight booking automatically approved?",
            "No. Bookings within 14 days require business justification and manager review."
        )
    ],

    "hotel_policy": [
        (
            "What is the hotel limit for executives?",
            "The executive hotel limit is INR 12,000 per night."
        ),
        (
            "What is the hotel limit for standard employees?",
            "The standard employee hotel limit is INR 6,000 per night."
        ),
        (
            "What is the hotel limit for contractors?",
            "The contractor hotel limit is INR 4,000 per night."
        ),
        (
            "Can a standard employee book a hotel for INR 5,000?",
            "Yes. INR 5,000 is within the standard employee limit of INR 6,000 per night."
        ),
        (
            "Can a contractor book a hotel for INR 5,000?",
            "INR 5,000 exceeds the contractor limit of INR 4,000. Manager approval is required for an exception."
        ),
        (
            "An executive wants a hotel costing INR 10,000. Is it within policy?",
            "Yes. INR 10,000 is within the executive limit of INR 12,000 per night."
        ),
        (
            "A standard employee selected a hotel costing INR 8,000. What should happen?",
            "INR 8,000 exceeds the standard employee limit of INR 6,000. Manager approval is required for an exception."
        ),
        (
            "What happens when hotel costs exceed the limit?",
            "The exception requires manager approval."
        ),
        (
            "Can a contractor book a hotel for INR 3,500?",
            "Yes. INR 3,500 is within the contractor limit of INR 4,000 per night."
        ),
        (
            "Is an INR 12,000 hotel within the standard employee limit?",
            "No. It exceeds the standard employee limit of INR 6,000 per night."
        )
    ],

    "meal_policy": [
        (
            "What is the executive daily meal allowance?",
            "The executive daily meal allowance is INR 2,500."
        ),
        (
            "What is the standard employee meal allowance?",
            "The standard employee daily meal allowance is INR 1,500."
        ),
        (
            "What is the contractor daily meal allowance?",
            "The contractor daily meal allowance is INR 1,000."
        ),
        (
            "Can a standard employee claim INR 1,200 for meals?",
            "Yes. INR 1,200 is within the standard employee daily limit of INR 1,500."
        ),
        (
            "Can a contractor claim INR 1,200 for meals?",
            "INR 1,200 exceeds the contractor daily limit of INR 1,000. Justification and finance review are required."
        ),
        (
            "Can an executive claim INR 2,000 for meals?",
            "Yes. INR 2,000 is within the executive daily limit of INR 2,500."
        ),
        (
            "A standard employee spends INR 1,800 on meals. What happens?",
            "INR 1,800 exceeds the standard employee limit of INR 1,500. Justification and finance review are required."
        ),
        (
            "What happens when meal expenses exceed the allowance?",
            "Amounts exceeding the applicable limit require justification and finance review."
        ),
        (
            "Can a contractor claim INR 900 per day?",
            "Yes. INR 900 is within the contractor daily limit of INR 1,000."
        ),
        (
            "Is INR 2,500 within the standard employee meal allowance?",
            "No. It exceeds the standard employee daily limit of INR 1,500."
        )
    ],

    "receipt_policy": [
        (
            "Are receipts mandatory for expenses above INR 1,000?",
            "Yes. Receipts are mandatory for individual expenses exceeding INR 1,000."
        ),
        (
            "Do I need a receipt for an INR 500 expense?",
            "The policy does not mandate a receipt for an individual expense of INR 500."
        ),
        (
            "I have an expense of INR 1,500 but no receipt. What happens?",
            "The receipt is mandatory because the expense exceeds INR 1,000. Missing receipts require finance review."
        ),
        (
            "Is a receipt required for INR 1,200?",
            "Yes. The individual expense exceeds INR 1,000."
        ),
        (
            "What happens if I lose a receipt?",
            "Missing receipts require finance review."
        ),
        (
            "Is a receipt mandatory for INR 1,000 exactly?",
            "The policy mandates receipts for amounts exceeding INR 1,000, not amounts equal to INR 1,000."
        ),
        (
            "Can finance review an expense with a missing receipt?",
            "Yes. The policy requires finance review for missing receipts."
        ),
        (
            "What is the receipt threshold?",
            "Receipts are mandatory for individual expenses exceeding INR 1,000."
        ),
        (
            "Do all expenses require receipts?",
            "The sample policy mandates receipts for individual expenses exceeding INR 1,000."
        ),
        (
            "I submitted an INR 2,000 expense without a receipt. Is it compliant?",
            "No. A receipt is mandatory for this expense, and missing receipts require finance review."
        )
    ],

    "mileage_policy": [
        (
            "What is the mileage reimbursement rate?",
            "The reimbursement rate is INR 12 per kilometre."
        ),
        (
            "Can I claim personal vehicle mileage?",
            "Yes, eligible personal vehicle travel can be reimbursed at INR 12 per kilometre."
        ),
        (
            "What is the reimbursement for 100 kilometres?",
            "The reimbursement is INR 1,200, subject to a valid business purpose and recorded distance."
        ),
        (
            "What is the reimbursement for 50 kilometres?",
            "The reimbursement is INR 600, subject to a valid business purpose and recorded distance."
        ),
        (
            "What is the reimbursement for 25 kilometres?",
            "The reimbursement is INR 300, subject to a valid business purpose and recorded distance."
        ),
        (
            "What information is needed for mileage reimbursement?",
            "A valid business purpose and travel distance must be recorded."
        ),
        (
            "Can I claim mileage without recording distance?",
            "No. The travel distance must be recorded."
        ),
        (
            "What is the claim for 200 kilometres?",
            "The reimbursement is INR 2,400, subject to a valid business purpose and recorded distance."
        ),
        (
            "Is the mileage rate INR 12 per kilometre?",
            "Yes. The sample policy specifies INR 12 per kilometre."
        ),
        (
            "Can personal vehicle business travel be reimbursed?",
            "Yes. Eligible personal vehicle travel is reimbursed at INR 12 per kilometre, with business purpose and distance recorded."
        )
    ],

    "approval_policy": [
        (
            "Who approves expenses up to INR 10,000?",
            "The Line Manager is the required approver."
        ),
        (
            "Who approves expenses of INR 15,000?",
            "The Line Manager followed by Finance must approve."
        ),
        (
            "Who approves expenses above INR 50,000?",
            "The Line Manager, Department Head and Finance must approve."
        ),
        (
            "Who approves an expense of INR 8,000?",
            "The Line Manager is the required approver."
        ),
        (
            "Who approves an expense of INR 30,000?",
            "The Line Manager followed by Finance must approve."
        ),
        (
            "Who approves an expense of INR 60,000?",
            "The Line Manager, Department Head and Finance must approve."
        ),
        (
            "Can the AI agent approve an exception?",
            "No. Final approval authority remains with designated human approvers."
        ),
        (
            "Who approves an expense of exactly INR 10,000?",
            "The Line Manager is the required approver."
        ),
        (
            "Who approves an expense of exactly INR 50,000?",
            "The Line Manager followed by Finance must approve."
        ),
        (
            "Can the AI override a hard policy restriction?",
            "No. Hard restrictions cannot be overridden by the AI agent."
        )
    ]
}


def build_dataset():

    records = []
    counter = 1

    for category, questions in DATASET.items():

        for question, gold_answer in questions:

            records.append({
                "id": f"T{counter:03d}",
                "category": category,
                "query": question,
                "gold_answer": gold_answer,
                "expected_result": "POLICY_LOOKUP"
            })

            counter += 1

    return records


if __name__ == "__main__":

    output = Path("data/evaluation/evaluation_dataset.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()

    with open(output, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    print(f"Dataset created successfully: {len(dataset)} queries")
    print(f"Saved to: {output}")

