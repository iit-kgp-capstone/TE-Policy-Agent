# T&E Policy Agent – Project Scope

**Document type:** Project Scope  
**Project:** Travel & Expense (T&E) Policy Agent  
**Status:** Initial project deliverable

## 1. Project Objective

Develop an AI-powered Travel and Expense Policy Copilot using a hosted instruction model with 8 billion parameters or fewer.

The system will retrieve company policies, validate travel and expense requests, identify policy violations, and recommend appropriate actions.

## 2. Core Responsibilities

The agent is responsible for:

1. Employee travel eligibility validation.
2. Flight booking policy validation.
3. Hotel accommodation limit validation.
4. Meal and per diem calculation.
5. Expense claim validation.
6. Receipt requirement verification.
7. Mileage reimbursement validation.
8. Approval hierarchy identification.
9. Policy exception handling.
10. Policy-based recommendations.

## 3. Agent Inputs

The agent may use the following employee, travel, and transaction information:

- Employee ID
- Employee grade
- Department
- Travel destination
- Travel dates
- Expense category
- Expense amount
- Receipt information
- Project code

The agent must identify missing required information rather than assume values.

## 4. Agent Outputs

For each request, the agent should provide the applicable information, where supported by policy evidence:

- Policy compliance status
- Applicable policy reference
- Allowed expense amount
- Violation category
- Violation explanation
- Recommended action
- Required approval level

## 5. Policy Decision Categories

Every evaluated request must be assigned an appropriate decision category.

| Category | Meaning |
|---|---|
| `PASS` | The expense or request complies with the applicable policy. |
| `SOFT_FLAG` | The request requires justification or additional review. |
| `HARD_FLAG` | The request violates a mandatory restriction. |
| `NEEDS_INFORMATION` | Required employee or transaction information is missing. |
| `HUMAN_REVIEW` | The policy is ambiguous or the decision requires authorized human approval. |

The agent must explain the reason for its classification and provide supporting policy references whenever available.

## 6. What the Agent Must Refuse

The agent must not:

- Invent policy limits.
- Fabricate policy citations.
- Approve unauthorized expenses.
- Override hard policy restrictions.
- Make unsupported reimbursement commitments.
- Treat missing information as approval.

When reliable policy evidence is unavailable, the agent must clearly state that limitation and request the required information or escalate the case for human review.

## 7. Technical Constraints

The implementation must follow these constraints:

- Hosted instruction model with 8 billion parameters or fewer.
- No direct model weight access.
- No model fine-tuning or weight modification.
- OpenAI-compatible chat completion API.
- Hosted embeddings endpoint.
- Retrieval-Augmented Generation (RAG).
- Tool-based agent execution.
- Measurable evaluation and observability.

The model is accessed through hosted APIs; the project does not download or modify model weights.

## 8. Human-in-the-Loop

All exceptions requiring managerial or finance authorization must be escalated to an authorized reviewer.

The AI agent can analyze a request and recommend an action, but it cannot substitute for an approval signature.

Human authorization remains mandatory wherever the applicable policy or approval hierarchy requires it.

## 9. Scope and Decision-Making Principles

The agent's decisions must be grounded in retrieved company policy and the employee or transaction information supplied.

The agent should:

1. Retrieve relevant policy evidence before making a policy-based determination.
2. Distinguish policy requirements from recommendations.
3. Explain the reasoning behind compliance classifications.
4. Identify missing or conflicting information.
5. Escalate ambiguous cases and approval-dependent exceptions.
6. Avoid claiming that an expense is reimbursable unless the applicable policy supports that conclusion.

## 10. Initial Deliverable

This document establishes the initial project scope and serves as the reference for subsequent project deliverables, including:

- System architecture and component design.
- Policy document ingestion and retrieval.
- Embedding and vector search integration.
- LLM client and agent implementation.
- Policy validation tools and decision logic.
- Human approval and exception workflows.
- Evaluation, testing, and observability.

