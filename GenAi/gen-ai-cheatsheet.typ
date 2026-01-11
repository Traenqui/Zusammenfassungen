#import "../template_zusammenf.typ": *
#import "@preview/wrap-it:0.1.1": wrap-content

#show: project.with(
  authors: ("Jonas Gerber"),
  fach: "GenAI",
  fach-long: "Generative AI",
  semester: "HS25",
  language: "en",
  column-count: 4,
  font-size: 4pt,
  landscape: true,
)

#include "src/01-deep-neural-network.typ"
#include "src/02-transformer.typ"
#include "src/03-large-language-models.typ"
#include "src/04-fine-tuning.typ"
#include "src/05-retrieval-augmented-generation.typ"
#include "src/06-agent.typ"
#include "src/07-llm-evaluation.typ"
#include "src/08-autoencoder.typ"
#include "src/09-vision-transformers.typ"
#include "src/10-diffusion.typ"
