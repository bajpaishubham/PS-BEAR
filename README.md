# American Black PS-BEAR

American Black PS-BEAR is a quick prototype built with the help of generative AI models. Although, the Searcher and Solver and hence the visible application do not use any generative AI models, subsequent versions and other PS-BEARs will be built mainly with the help of deterministic models. The application is live at [this url](https://psbear.streamlit.app/).

**Note:** Default copyright laws apply, meaning that the repository owner retains all rights to the source code and no one may reproduce, distribute, or create derivative works from this work.

**v1.0.0:** For this version, Claude Code with model `Opus 4.8` was used to build the automated batch data processing pipeline and the Solver and Searcher sections. For description generation for the various topics and subtopics and for generation of the formulae table from formulae present in [Problems in General Physics by I.E. Irodov](https://books.google.co.in/books/about/Problems_in_General_Physics.html?id=Qj0L0QEACAAJ&redir_esc=y), `gemini-2.5-flash` model and Mistral OCR (`mistral-ocr-4-launch`) was used.

**TODO List**
- [ ] Image-wise data processing and manual validation and updation of data processed
- [ ] Wikipedia (or some other credible source) based descriptions (basically not AI generated)
- [ ] Multi-formula approach
- [ ] Simulator based solutions

## Acknowledgement

If you refer to this software or work in an academic publication, please cite as below:

```
@unpublished{bajpai2024psbear,
   author={Bajpai, Shubham},
   title={Automated approaches for solving physics problems},
   doi={10.1109/LRA.2023.3270034},
   year=2024
}
```
