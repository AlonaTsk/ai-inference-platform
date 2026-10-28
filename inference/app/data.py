"""Synthetic support-ticket dataset.

The tickets are generated from templates, so this is NOT real customer data.
To keep the task from being trivially separable, the generator:
  * shares generic openers/closers across every class,
  * injects "distractor" phrases from a different class into some tickets,
  * flips a small fraction of labels to simulate annotation noise.
Everything is driven by a seeded RNG, so the dataset is fully reproducible.
"""

import random

LABELS = ["account", "billing", "shipping", "technical"]

OPENERS = [
    "Hi,", "Hello team,", "Hey,", "Good morning,", "", "", "Urgent:",
    "Please help.", "Not sure who to ask, but", "Following up on my last message,",
]

CLOSERS = [
    "Thanks.", "Thank you in advance.", "Please advise.", "", "", "",
    "This is really frustrating.", "Can someone look at this today?",
    "Order number is in my profile.", "Regards",
]

CORE = {
    "account": [
        "I can't log in to my account",
        "the password reset email never arrives",
        "my account got locked after too many attempts",
        "I want to change the email address on my profile",
        "two-factor authentication codes are not accepted",
        "please delete my account and personal data",
        "I can't verify my phone number",
        "someone else seems to have accessed my account",
        "how do I update my username",
        "my profile shows the wrong name",
    ],
    "billing": [
        "I was charged twice this month",
        "please refund my last payment",
        "my invoice shows the wrong amount",
        "the discount code was not applied at checkout",
        "I want to cancel my subscription before renewal",
        "my credit card was declined but money was taken",
        "how do I download a VAT receipt",
        "the price went up without notice",
        "I need to change my payment method",
        "the trial converted to a paid plan without warning",
    ],
    "shipping": [
        "my package has not arrived yet",
        "the tracking number does not work",
        "the parcel was delivered to the wrong address",
        "the box arrived damaged",
        "can I change the delivery address for my order",
        "the courier marked it delivered but I have nothing",
        "delivery is two weeks late",
        "an item was missing from the shipment",
        "do you ship to Norway",
        "I want to return the item I received",
    ],
    "technical": [
        "the app crashes when I open settings",
        "the page keeps loading forever",
        "I get a 500 error when saving",
        "the export to CSV produces an empty file",
        "notifications stopped working after the update",
        "the search returns no results",
        "the dashboard is blank in Safari",
        "file upload fails with a timeout",
        "the API returns 401 even with a valid key",
        "the mobile app freezes on startup",
    ],
}

DETAILS = {
    "account": ["since yesterday", "on the web and on mobile", "I already tried incognito mode",
                "it says my session expired", "my login worked last week"],
    "billing": ["on my card statement", "for the annual plan", "the amount was 49 euros",
                "according to my bank", "for last month's invoice"],
    "shipping": ["the order was placed ten days ago", "the courier is not answering",
                 "the estimated date has passed", "for order 12345", "it was supposed to be express"],
    "technical": ["on version 4.2", "I cleared the cache already", "it happens every time",
                  "the console shows an error", "on Windows and Mac"],
}


def _ticket(rng: random.Random, label: str, distractor_rate: float) -> str:
    parts = [rng.choice(OPENERS), rng.choice(CORE[label])]
    if rng.random() < 0.6:
        parts.append(rng.choice(DETAILS[label]))
    if rng.random() < distractor_rate:
        other = rng.choice([l for l in LABELS if l != label])
        parts.append("also " + rng.choice(DETAILS[other] + CORE[other][:3]))
    parts.append(rng.choice(CLOSERS))
    return " ".join(p for p in parts if p).strip()


def generate(n: int = 4000, seed: int = 42, distractor_rate: float = 0.35,
             label_noise: float = 0.05) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        texts.append(_ticket(rng, label, distractor_rate))
        if rng.random() < label_noise:
            label = rng.choice([l for l in LABELS if l != label])
        labels.append(label)
    return texts, labels
