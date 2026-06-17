# Software development

Source: https://en.wikipedia.org/wiki/Software_development
Retrieved at: 2026-06-17T17:12:34Z
Revision ID: 1358647165
Raw text SHA256: 1e06939e0681a92bff3fb35ed2e721c8a7c8e7dfef07920ef413b3d2e5789aec
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Software development is the process of designing, creating, testing, and maintaining software applications to meet specific user needs or business objectives. The process is more encompassing than programming, writing code, because it includes conceiving the goal, evaluating feasibility, analyzing requirements, design, testing and release. The process is part of software engineering which also includes organizational management, project management, configuration management and other aspects.
Software development involves many skills and job specializations, including programming, testing, documentation, graphic design, user support, marketing, and fundraising. Common types of tools are compilers, integrated development environments (IDEs), and version control.
The details of the process used for a development effort vary. The process may be confined to a formal, documented standard, or it can be customized and emergent for the development effort. The process may be sequential, in which each major phase (i.e., design, implement, and test) is completed before the next begins, but an iterative approach—where small aspects are separately designed, implemented, and tested—can reduce risk and cost and increase quality.

Methodologies

Each of the available methodologies is best suited to specific kinds of projects, based on various technical, organizational, project, and team considerations.

The simplest methodology is the "code and fix", typically used by a single programmer working on a small project. After briefly considering the purpose of the program, the programmer codes it and runs it to see if it works. When they are done, the product is released. This methodology is useful for prototypes but cannot be used for more elaborate programs.
In the top-down waterfall model, feasibility, analysis, design, development, quality assurance, and implementation occur sequentially in that order. This model requires one step to be complete before the next begins, causing delays, and makes it impossible to revise previous steps if necessary.
With iterative processes these steps are interleaved with each other for improved flexibility, efficiency, and more realistic scheduling. Instead of completing the project all at once, one might go through most of the steps with one component at a time. Iterative development also lets developers prioritize the most important features, enabling lower priority ones to be dropped later on if necessary. Agile is one popular method, originally intended for small or medium sized projects, that focuses on giving developers more control over the features that they work on to reduce the risk of time or cost overruns. Derivatives of agile include extreme programming and Scrum. Open-source software development typically uses agile methodology with concurrent design, coding, and testing, due to reliance on a distributed network of volunteer contributors.
Beyond agile, some companies integrate information technology (IT) operations with software development, which is called DevOps or DevSecOps including computer security. DevOps includes continuous development, testing, integration of new code in the version control system, deployment of the new code, and sometimes delivery of the code to clients. The purpose of this integration is to deliver IT services more quickly and efficiently.
Another focus in many programming methodologies is the idea of trying to catch issues such as security vulnerabilities and bugs as early as possible (shift-left testing) to reduce the cost of tracking and fixing them.
In 2009, it was estimated that 32% of software projects were delivered on time and on budget, and with full functionality. An additional 44% were delivered, but were missing at least one of their features. The remaining 24% were cancelled before release.
AI coding tools have become a standard part of how software gets built. Stack Overflow's 2025 Developer Survey found that 84% of professional developers use or plan to use AI tools — up from 76% the year before — with code generation and debugging as the top use cases. JetBrains put the figure at 85% in their 2025 developer survey. One tension that's emerged: trust hasn't kept pace with adoption. Stack Overflow found developer confidence in AI-generated code dropped from 40% in 2023 to 29% in 2025, even as usage kept climbing.

Life cycle
Software development life cycle describes the typical phases of the process of developing software.

Feasibility
The sources of ideas for software products are plentiful. These ideas can come from market research, including the demographics of potential new customers, existing customers, sales prospects who rejected the product, other internal software development staff, or a creative third party. Ideas for software products are usually first evaluated by marketing personnel for economic feasibility, fit with existing channels of distribution, possible effects on existing product lines, required features, and fit with the company's marketing objectives. In the marketing evaluation phase, the cost and time assumptions are evaluated.  The feasibility analysis estimates the project's return on investment, its development cost and its timeframe. Based on this analysis, the company can make a business decision to invest in further development.  After deciding to develop the software, the company is focused on delivering the product at or below the estimated cost and time and with a high standard of quality (i.e., lack of bugs) and the desired functionality. Nevertheless, most software projects run late, and sometimes compromises are made in features or quality to meet a deadline.

Analysis
Software analysis begins with a requirements analysis to capture the business needs of the software. Challenges for the identification of needs are that current or potential users may have different and incompatible needs, may not understand their own needs, and change their needs during the process of software development. Ultimately, the result of analysis is a detailed specification for the product that developers can work from. Software analysts often decompose the project into smaller objects, components that can be reused for increased cost-effectiveness, efficiency, and reliability. Decomposing the project may enable a multi-threaded implementation that runs significantly faster on multiprocessor computers.
During the analysis and design phases of software development, structured analysis is often used to break down the customer's requirements into pieces that can be implemented by software programmers. The underlying logic of the program may be represented in data-flow diagrams, data dictionaries, pseudocode, state transition diagrams, and/or entity relationship diagrams. If the project incorporates a piece of legacy software that has not been modeled, this software may be modeled to help ensure it is correctly incorporated with the newer software.

Design

Design involves choices about the implementation of the software, such as which programming languages and database software to use, or how the hardware and network communications will be organized. Design may be iterative with users consulted about their needs in a process of trial and error. Design often involves people who are expert in aspects such as database design, screen architecture, and the performance of servers and other hardware. Designers often attempt to find patterns in the software's functionality to spin off distinct modules that can be reused with object-oriented programming. An example of this is the model–view–controller, an interface between a graphical user interface and the backend.

Programming

The central feature of software development is creating and understanding the software that implements the desired functionality. There are various strategies for writing the code. Cohesive software has various components that are independent from each other. Coupling is the interrelation of different software components, which is viewed as undesirable because it increases the difficulty of maintenance. Often, software programmers do not follow industry best practices, resulting in code that is inefficient, difficult to understand, or lacking documentation on its functionality. These standards are especially likely to break down in the presence of deadlines. As a result, testing, debugging, and revising the code become much more difficult. Code refactoring is a technique for restructuring existing code without changing its external behavior, often to improve its design, readability, or maintainability.
Since the popularization of large language models, AI-assisted software development has been used to augment human programming by having an AI handle syntax and write code.

Testing

Testing is the process of ensuring that the code executes correctly and without errors. Debugging is performed by each software developer on their own code to confirm that the code does what it is intended to. In particular, it is crucial that the software executes on all inputs, even if the result is incorrect. Code reviews by other developers are often used to scrutinize new code added to the project, and according to some estimates dramatically reduce the number of bugs persisting after testing is complete. Once the code has been submitted, quality assurance – a separate department of non-programmers for most large companies – test the accuracy of the entire software product. Testing activities may also occur throughout the software development life cycle, depending on the methodology used. Acceptance tests derived from the original software requirements are a popular tool for this. Quality testing also often includes stress and load checking (whether the software is robust to heavy levels of input or usage), integration testing (to ensure that the software is adequately integrated with other software), and compatibility testing (measuring the software's performance across different operating systems or browsers). When tests are written before the code, this is called test-driven development.

Production

Production is the phase in which software is deployed to the end user. During production, the developer may create technical support resources for users or a process for fixing bugs and errors that were not caught earlier. There might also be a return to earlier development phases if user needs changed or were misunderstood.

Workers

Software development is performed by software developers, usually working on a team. Efficient communications between team members is essential to success. This is more easily achieved if the team is small, used to working together, and located near each other. Communications also help identify problems at an earlier stage of development and avoid duplicated effort. Many development projects avoid the risk of losing essential knowledge held by only one employee by ensuring that multiple workers are familiar with each component. Software development involves professionals from various fields, not just software programmers but also product managers who set the strategy and roadmap for the product, individuals specialized in testing, documentation writing, graphic design, user support, marketing, and fundraising. Although workers for proprietary software are paid, most contributors to open-source software are volunteers. Alternately, they may be paid by companies whose business model does not involve selling the software, but something else – such as services and modifications to open source software.

Models and tools

Computer-aided software engineering
Computer-aided software engineering (CASE) is tools for the partial automation of software development. CASE enables designers to sketch out the logic of a program, whether one to be written, or an already existing one to help integrate it with new code or reverse engineer it (for example, to change the programming language).

Documentation

Documentation comes in two forms that are usually kept separate – one intended for software developers, and another made available to the end user to help them use the software. Most developer documentation is in the form of code comments for each file, class, and method that cover the application programming interface (API)—how the piece of software can be accessed by another—and often implementation details. This documentation is helpful for new developers to understand the project when they begin working on it. In agile development, the documentation is often written at the same time as the code. User documentation is more frequently written by technical writers.

Effort estimation

Accurate estimation is crucial at the feasibility stage and in delivering the product on time and within budget. The process of generating estimations is often delegated by the project manager. Because the effort estimation is directly related to the size of the complete application, it is strongly influenced by the addition of features in the requirements—the more requirements, the higher the development cost. Aspects not related to functionality, such as the experience of the software developers and code reusability, are also essential to consider in estimation. As of 2019, most of the tools for estimating the amount of time and resources for software development were designed for conventional applications and are not applicable to web applications or mobile applications.

Integrated development environment

An integrated development environment (IDE) supports software development with enhanced features compared to a simple text editor. IDEs often include automated compiling, syntax highlighting of errors, debugging assistance, integration with version control, and semi-automation of tests.

Version control

Version control is a popular way of managing changes made to the software. Whenever a new version is checked in, the software saves a backup of all modified files. If multiple programmers are working on the software simultaneously, it manages the merging of their code changes. The software highlights cases where there is a conflict between two sets of changes and allows programmers to fix the conflict.

View model

A view model is a framework that provides the viewpoints on the system and its environment, to be used in the software development process. It is a graphical representation of the underlying semantics of a view.
The purpose of viewpoints and views is to enable human engineers to comprehend very complex systems and to organize the elements of the problem around domains of expertise. In the engineering of physically intensive systems, viewpoints often correspond to capabilities and responsibilities within the engineering organization.

Fitness functions
Fitness functions are automated and objective tests to ensure that the new developments do not deviate from the established constraints, checks and compliance controls.

Intellectual property

Intellectual property can be an issue when developers integrate open-source code or libraries into a proprietary product, because most open-source licenses used for software require that modifications be released under the same license. As an alternative, developers may choose a proprietary alternative or write their own software module.

See also
List of AI-assisted software development tools
List of software developed at universities
AI-assisted software development

References

Further reading

External links
Media related to Software development at Wikimedia Commons
