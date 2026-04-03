import json
from neo4j_loader import Neo4jLoader
from llm_extractor import Triple

data = {
"triples": [
{"subject": "Palantir", "predicate": "HAS_SEGMENT", "object": "Government Segment", "subject_type": "Company", "object_type": "BusinessSegment"},
{"subject": "Palantir", "predicate": "HAS_SEGMENT", "object": "Commercial Segment", "subject_type": "Company", "object_type": "BusinessSegment"},
{"subject": "Palantir", "predicate": "OFFERS", "object": "Palantir Gotham", "subject_type": "Company", "object_type": "Offering"},
{"subject": "Palantir", "predicate": "OFFERS", "object": "Palantir Foundry", "subject_type": "Company", "object_type": "Offering"},
{"subject": "Palantir", "predicate": "OFFERS", "object": "Palantir Apollo", "subject_type": "Company", "object_type": "Offering"},
{"subject": "Palantir", "predicate": "OFFERS", "object": "Palantir AIP", "subject_type": "Company", "object_type": "Offering"},
{"subject": "Palantir", "predicate": "OFFERS", "object": "Developer Tier", "subject_type": "Company", "object_type": "Offering"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Utility Operations Analysts", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Automotive Manufacturing Workers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Oil and Gas Technicians and Operators", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Pharmaceutical Researchers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Supply-Chain Managers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Public Health Administrators", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Special Forces Personnel", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Military Officials", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Government Customers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Commercial Customers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Intelligence Community", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Global Defense Agencies", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Disaster Relief Organizations", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Law Enforcement Agencies", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Financial Institutions", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Telecommunications Companies", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "SERVES", "object": "Clinical Researchers", "subject_type": "Company", "object_type": "CustomerType"},
{"subject": "Palantir Gotham", "predicate": "SERVES", "object": "Global Defense Agencies", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Gotham", "predicate": "SERVES", "object": "Intelligence Community", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Gotham", "predicate": "SERVES", "object": "Disaster Relief Organizations", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Gotham", "predicate": "SERVES", "object": "Government Customers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Foundry", "predicate": "SERVES", "object": "Commercial Customers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Foundry", "predicate": "SERVES", "object": "Government Customers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Foundry", "predicate": "SERVES", "object": "Financial Institutions", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir Foundry", "predicate": "SERVES", "object": "Telecommunications Companies", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir AIP", "predicate": "SERVES", "object": "Commercial Customers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir AIP", "predicate": "SERVES", "object": "Government Customers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Developer Tier", "predicate": "SERVES", "object": "Developers", "subject_type": "Offering", "object_type": "CustomerType"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "United States", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "South Korea", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "United Kingdom", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "Scandinavia", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "Northern Europe", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "Europe", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "France", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "OPERATES_IN", "object": "Japan", "subject_type": "Company", "object_type": "Place"},
{"subject": "Palantir", "predicate": "SELLS_THROUGH", "object": "Direct Sales Force", "subject_type": "Company", "object_type": "Channel"},
{"subject": "Palantir", "predicate": "SELLS_THROUGH", "object": "Channel Sales", "subject_type": "Company", "object_type": "Channel"},
{"subject": "Palantir", "predicate": "SELLS_THROUGH", "object": "Cloud Partnerships", "subject_type": "Company", "object_type": "Channel"},
{"subject": "Palantir", "predicate": "SELLS_THROUGH", "object": "Joint Ventures", "subject_type": "Company", "object_type": "Channel"},
{"subject": "Palantir", "predicate": "PARTNERS_WITH", "object": "Fujitsu Limited", "subject_type": "Company", "object_type": "Company"}
]
}

mapped_triples = []
for t in data["triples"]:
    mapped_triples.append(Triple(
        subject=t["subject"],
        subject_type=t["subject_type"],
        relation=t["predicate"], # Map predicate back to the 'relation' pydantic schema key
        object=t["object"],
        object_type=t["object_type"]
    ))

print(f"Mapped {len(mapped_triples)} raw triples into Python objects.")
loader = Neo4jLoader()
loader.load_triples(mapped_triples)
print("Insertion complete!")
