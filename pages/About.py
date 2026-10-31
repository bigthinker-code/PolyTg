
import streamlit as st

st.set_page_config(
    page_title="About | PolyTg",
    page_icon="🧪",
    layout="wide"
)

st.title("About PolyTg")

st.write(
    "An introduction to the project, its motivation "
    "and the person behind its development."
)

st.divider()

st.header("The project")

st.markdown("""
PolyTg is a machine-learning platform designed to predict
the glass transition temperature (Tg) of polymers from
their polymer SMILES representations.

The project explores how computational chemistry,
polymer descriptors and machine learning can be combined
to support polymer property estimation.

Beyond generating predictions, PolyTg incorporates
uncertainty estimation and applicability-domain assessment
to encourage more informed interpretation of its results.
""")

st.divider()

st.header("Why glass transition temperature?")

st.write(
    "Glass transition temperature is an important property "
    "in polymer science. It provides insight into the "
    "temperature-dependent behaviour of amorphous polymer "
    "materials and can influence their processing and "
    "potential applications."
)

st.write(
    "Experimental determination can require time and "
    "resources. Computational prediction offers a way "
    "to explore polymer properties and support early-stage "
    "materials research."
)

st.divider()

st.header("Technology")

technologies = [
    ("Python", "Core programming language"),
    ("PolyMetriX", "Polymer representation and descriptor generation"),
    ("RDKit", "Cheminformatics functionality"),
    ("Scikit-learn", "ExtraTrees machine-learning model"),
    ("Hugging Face", "Trained model artifact hosting"),
    ("Streamlit", "Interactive application and deployment"),
]

for name, description in technologies:
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(description)

st.divider()

st.header("Developer")

st.subheader("Agboola Abdulroheem Oyindamola")

st.markdown("**Machine Learning Engineer**")

st.write(
    "PolyTg was developed as an independent computational "
    "research and machine-learning project exploring the "
    "application of artificial intelligence to polymer science."
)

st.caption(
    "Interested in machine learning, computational chemistry, "
    "cheminformatics and data-driven scientific research."
)

st.divider()

st.header("Project links")

st.markdown(
    "[GitHub Repository](https://github.com/bigthinker-code/PolyTg)"
)

st.markdown(
    "[Trained Model on Hugging Face]"
    "(https://huggingface.co/Abdulroheem/PolyTg)"
)

st.markdown(
    "[Live Prediction Application]"
    "(https://polytg-7jp38fvdkcauywo6eafghb.streamlit.app/)"
)

st.divider()

st.caption(
    "PolyTg | Independent Research Prototype"
)
