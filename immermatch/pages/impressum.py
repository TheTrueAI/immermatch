"""Impressum / Legal Notice — required by § 5 DDG (Digitale-Dienste-Gesetz)."""

import contextlib
import os

import streamlit as st

REQUIRED_KEYS = ("IMPRESSUM_NAME", "IMPRESSUM_ADDRESS", "IMPRESSUM_EMAIL")

for key in REQUIRED_KEYS:
    if key not in os.environ:
        with contextlib.suppress(KeyError, FileNotFoundError):
            os.environ[key] = st.secrets[key]

_name = os.environ.get("IMPRESSUM_NAME", "")
_address = os.environ.get("IMPRESSUM_ADDRESS", "")
_email = os.environ.get("IMPRESSUM_EMAIL", "")

st.set_page_config(page_title="Immermatch – Legal Notice", page_icon="⚖️")

st.title("Legal Notice / Impressum")
st.caption("Information pursuant to § 5 DDG (Digitale-Dienste-Gesetz)")

if not all((_name, _address, _email)):
    required_keys_text = ", ".join(REQUIRED_KEYS[:-1]) + f", and {REQUIRED_KEYS[-1]}"
    st.warning(f"Impressum contact details are not configured. Set {required_keys_text}.")

st.markdown(f"""
**{_name}**

{_address}

Email: {_email}

---

### Disclaimer

#### Liability for Content
The contents of our pages have been created with the utmost care. However, we
cannot guarantee the accuracy, completeness, or timeliness of the content. As a
service provider, we are responsible for our own content on these pages under
general law pursuant to § 7(1) DDG. According to §§ 8–10 DDG, however, we are
not obligated to monitor transmitted or stored third-party information or to
investigate circumstances indicating unlawful activity.

#### Liability for Links
Our website contains links to external third-party websites over whose content
we have no influence. We therefore cannot accept any liability for this
third-party content. The respective provider or operator of the linked pages is
always responsible for the content of those pages.
""")
