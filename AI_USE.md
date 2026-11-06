## Below are the prompts given to ChatGPT, which helped us generating this report.

1) 
HERE IS MY INTRODUCTION POLISH THE GRAMMAR AND WRITE BETTER: Large Language Models (LLMs) are becoming more widely used to provide information in different areas, including healthcare and medicine. However, they can also create safety risks, especially when users ask for dangerous medical instructions or harmful advice. Giving this type of information can have serius consequences, which makes it important for models to know when they shourefuse a request.Even though LLMs have safety mechnisms to prevent harmful responses, they do not always make the right decision. Sometimes they answer dangerous medical questions that should be refused, while other times they refuse harmless questions that could be answered safely. This is known as over-refusal, and it can make models less useful by preventing users from getting information they actually need.The main challange is finding a balance between safety and usefulness. Models should be able to refuse dangerous requests without refusing normal medical questions. This project focuses on improving topic-specific refusal in LLMs, particularly their ability to distinguish between dangerous medical prompts and harmless medical questions. The aim is to improve refusal behavior while still allowing models to provide useful medical information when it is safe to do so.

2) 
TELL ME THE STRENGHTS AND WEAKNESS OF THIS LITERATURE REVIEW I DID A BRAIN DUMP O WHAT I KNEW:
Finetuning is a fragile way of chnaging llm behavious ars these data sets ar every limited and can lead to various problems like racism sexism etc as shown by \cite{betley2025weird}. Steerring comes out to be a cheap way to chnage the LLm behavior on inference time as re training llms are very costly. We limit our study to refusal in autoregresive llms LLama 3.1B, we only focus on refusal because it is well studied and has a singular universal direction as shown by  \cite{arditi2024refusal}, which shows the difference of means appraoch to get the direction. There are various studies to remove done on various types of activation steering such as CLAS \cite{hsu2026contextual} \cite{turner2023steering} however we find very less attempts which focus on  doing refusal on a particular topic like medical in our case. 

3)
explain activation steering and contextual linear activation steering and write it's equations and math in latex. Before writing anything refer to the sources given:

@article{hsu2026contextual,
  title={Contextual Linear Activation Steering of Language Models},
  author={Hsu, Brandon and Beaglehole, Daniel and Radhakrishnan, Adityanarayanan and Belkin, Mikhail},
  journal={arXiv preprint arXiv:2604.24693},
  year={2026}
}

@article{arditi2024refusal,
  title={Refusal in language models is mediated by a single direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={136037--136083},
  year={2024}
}

4) My report is comming together nicely, but I think has some useless info what are some of the standards of writing I should use so that the report is to the point here is the report tell me.

5) make the table attached
in this latex style \centering
\begin{tabular}{lll}
\toprule
\multicolumn{2}{c}{Part}                   \\
\cmidrule(r){1-2}
Name     & Description     & Size ($\mu$m) \\
\midrule
Dendrite & Input terminal  & $\sim$100     \\
Axon     & Output terminal & $\sim$10      \\
Soma     & Cell body       & up to $10^6$  \\
\bottomrule
\end{tabular}






