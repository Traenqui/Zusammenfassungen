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

#let def(term, body) = [_{term}_: {body}]

= Core NN

= Transformer
