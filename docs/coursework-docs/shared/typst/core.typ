#let config = yaml("../../term-paper.yaml")

#let project = config.project
#let student = config.student
#let students = config.at("students", default: ((name: student.name, group: student.group),))
#let supervisor = config.supervisor
#let approver = config.approver
#let organization = config.organization

#let faculty-name = "Факультет компьютерных наук"
#let program-name = "Образовательная программа \"Программная инженерия\""
#let current-year = datetime.today().year()

#let project-name() = project.name
#let project-name-english() = project.english_name
#let project-summary() = project.summary
#let project-type() = project.type

#let display-project-name(id) = project.name

#let document-title(id) = if id == "tz" {
  "Техническое задание"
} else if id == "pz" {
  "Пояснительная записка"
} else if id == "pmi" or id == "pmi-team" {
  "Программа и методика испытаний"
} else if id == "ro" {
  "Руководство оператора"
} else {
  "Документ"
}

#let document-code(id) = if id == "tz" {
  "RU.17701729." + project.code + " ТЗ 01-1"
} else if id == "pz" {
  "RU.17701729." + project.code + " 81 01-1"
} else if id == "pmi" or id == "pmi-team" {
  "RU.17701729." + project.code + " 51 01-1"
} else if id == "ro" {
  "RU.17701729." + project.code + " 34 01-1"
} else {
  "RU.17701729." + project.code
}

#let approval-code(id) = document-code(id) + "-ЛУ"

#let paragraph(body) = {
  h(2em)
  body
}

#let document-annotation(id) = if id == "tz" [
  Техническое задание -- это основной документ, оговаривающий набор требований и порядок создания программного продукта, в соответствии с которым производится разработка программы, ее тестирование и приемка.

  Настоящее Техническое задание на разработку "#project-name()" содержит следующие разделы: "Введение", "Основания для разработки", "Назначение разработки", "Требования к программе", "Требования к программной документации", "Технико-экономические показатели", "Стадии и этапы разработки", "Порядок контроля и приемки", приложения [7].

  Настоящий документ разработан в соответствии с требованиями ГОСТ 19.101-2024 [1], ГОСТ 19.102-77 [2], ГОСТ 19.103-77 [3], ГОСТ 19.104-78 [4], ГОСТ 19.105-78 [5], ГОСТ 19.106-78 [6], ГОСТ 19.201-78 [7].
 ] else if id == "pz" [
  Пояснительная записка содержит сведения о назначении программы, области ее применения, технических характеристиках, выбранных средствах реализации, архитектурных решениях и ожидаемых технико-экономических показателях разработки.

  Документ подготовлен в составе программной документации курсового проекта "#display-project-name(id)" в соответствии с требованиями Единой системы программной документации.
] else if id == "pmi" or id == "pmi-team" [
  Программа и методика испытаний определяет объект, цель, средства, порядок и методы испытаний программного продукта "#display-project-name(id)".

  Документ предназначен для проверки соответствия программы требованиям технического задания и фиксации критериев приемки результата разработки.
] else if id == "ro" [
  Руководство оператора содержит сведения о назначении программы "#display-project-name(id)", условиях ее выполнения, порядке запуска, основных сценариях работы и действиях пользователя в исключительных ситуациях.
] else [
  Документ подготовлен в составе программной документации курсового проекта "#display-project-name(id)".
]

#let executors-for(id) = if id == "ro" or id == "pmi-team" {
  students
} else {
  ((name: student.name, group: student.group),)
}

#let executors-block(id) = align(right)[
  #set par(spacing: 1em)

  Исполнители:

  #for person in executors-for(id) [
    Студент группы #person.group

    #("_" * 13) / #person.name /

    "#("_" * 3)" #("_" * 15) #current-year г.

    #v(1em)
  ]
]

#let render-document(id, body) = {
  let un(n) = "_" * n
  let doc-code = document-code(id)
  let doc-title = document-title(id)
  let approval-page-code = approval-code(id)

  let storage-table = {
    set text(size: 10pt)
    place(
      bottom + left,
      dx: 5mm,
      dy: -10mm,
      rotate(
        -90deg,
        reflow: true,
        table(
          columns: (25mm, 35mm, 25mm, 25mm, 35mm),
          rows: (5mm, 7mm),
          align: center,
          [Инв.№ подп], [Подп. и дата], [Взам. инв.№], [Инв.№ дубл.], [Подп. и дата],
        ),
      ),
    )
  }

  let approval-page = {
    let top-banner = [
      #set par(spacing: 0.65em)

      #text(weight: "bold", organization.name)

      #faculty-name

      #program-name
    ]

    let approve-table = grid(
      columns: (1fr, 1fr),
      inset: (
        x: 10mm,
        y: 3mm,
      ),
      align: center,

      [
        СОГЛАСОВАНО

        #supervisor.position
      ],
      [
        УТВЕРЖДЕНО

        #approver.position
      ],

      box[
        #un(13) #supervisor.name

        "#un(3)" #un(13) #current-year г.
      ],
      [
        #un(13) #approver.name

        "#un(3)" #un(13) #current-year г.
      ],
    )

    let center-banner = [
      #set text(size: 14pt, weight: "bold")
      #set par(spacing: 2em)

      #par(spacing: 0.65em, upper(display-project-name(id)))

      #doc-title

      ЛИСТ УТВЕРЖДЕНИЯ

      #approval-page-code
    ]

    let student-info = executors-block(id)

    let bottom-banner = [
      #set text(weight: "bold")

      #current-year
    ]

    page(
      header: none,
      footer: none,
      margin: (
        left: 20mm,
        right: 10mm,
        top: 25mm,
        bottom: 15mm,
      ),
      foreground: storage-table,
    )[
      #set align(center)

      #grid(
        columns: (1fr),
        row-gutter: 1fr,
        top-banner,
        approve-table,
        center-banner,
        student-info,
        bottom-banner,
      )
    ]

    counter(page).update(1)
  }

  let title-page = {
    let top-banner = [
      #set align(left)
      #set par(spacing: 2em)

      УТВЕРЖДЕН

      #approval-page-code
    ]

    let center-banner = [
      #set text(size: 14pt, weight: "bold")
      #set par(spacing: 2em)

      #par(spacing: 0.65em, upper(display-project-name(id)))

      #doc-title

      #doc-code

      Листов #context { counter(page).final().at(0) }
    ]

    let title-student-info = if id == "ro" or id == "pmi-team" {
      executors-block(id)
    } else {
      none
    }

    let bottom-banner = [
      #set text(weight: "bold")

      #current-year
    ]

    page(
      header: none,
      footer: none,
      margin: (
        left: 20mm,
        right: 10mm,
        top: 25mm,
        bottom: 15mm,
      ),
      foreground: storage-table,
    )[
      #set align(center)

      #if title-student-info == none [
        #grid(
          columns: (1fr),
          row-gutter: 1fr,
          top-banner,
          center-banner,
          bottom-banner,
        )
      ] else [
        #grid(
          columns: (1fr),
          row-gutter: 1fr,
          top-banner,
          center-banner,
          title-student-info,
          bottom-banner,
        )
      ]
    ]
  }

  let list-registration-changes = {
    let list-name = [
      #set text(size: 14pt, weight: "bold")
      #set par(spacing: 2em)
      ЛИСТ РЕГИСТРАЦИИ ИЗМЕНЕНИЙ
    ]

    let changes-table = figure(
      table(
        columns: (10mm,) + (15mm,) * 4 + (20mm,) * 2 + (auto,) + (auto,) * 2,
        rows: (auto,) * 3 + (9.5mm,) * 21,

        table.cell(colspan: 10, align: horizon, [Лист регистрации изменений]),
        table.cell(colspan: 5, align: horizon, [Номера листов (страниц)]),
        table.cell(rowspan: 2, align: horizon, [Всего листов (страниц в докум.)]),
        table.cell(rowspan: 2, align: horizon, [№ документа]),
        table.cell(rowspan: 2, align: horizon, [Входящий № сопроводительного докум. и дата]),
        table.cell(rowspan: 2, align: horizon, [Подп.]),
        table.cell(rowspan: 2, align: horizon, [Дата]),
        table.cell(align: horizon, rotate(-90deg, reflow: true)[Изм.]),
        table.cell(align: horizon, rotate(-90deg, reflow: true)[Измененных]),
        table.cell(align: horizon, rotate(-90deg, reflow: true)[Замененных]),
        table.cell(align: horizon, rotate(-90deg, reflow: true)[Новых]),
        table.cell(align: horizon, rotate(-90deg, reflow: true)[Аннулированных]),
      ),
    )

    page(
      header: none,
      footer: none,
      margin: (
        left: 20mm,
        right: 10mm,
        top: 25mm,
        bottom: 15mm,
      ),
    )[
      #set align(center)

      #grid(
        columns: (1fr),
        row-gutter: 1fr,
        list-name,
        changes-table,
      )
    ]
  }

  let outline-block = {
    pagebreak(weak: true)

    {
      set align(center)
      set text(weight: "bold")

      [СОДЕРЖАНИЕ]
    }

    outline(
      title: none,
      indent: 5mm,
    )
  }

  let normal-pages = {
    set page(
      margin: (
        top: 25mm,
        left: 20mm,
        right: 10mm,
        bottom: 47mm,
      ),
      header: [
        #set align(center)
        #set text(weight: "bold")

        #context counter(page).display()

        #doc-code
      ],
      footer: [
        #table(
          columns: (2fr, 1fr, 1fr, 1fr, 1fr),
          align: center,
          rows: 7mm,

          [], [], [], [], [],
          [Изм.], [Лист], [№ докум.], [Подп.], [Дата],
          doc-code, [], [], [], [],
          [Инв. № подл.], [Подп. и дата], [Взам. Инв. №], [Инв. № дубл.], [Подп. и дата],
        )
      ],
    )

    set par(
      first-line-indent: (
        amount: 2em,
        all: true,
      ),
      justify: true,
      spacing: 0.65em,
      leading: 0.65em,
    )

    set list(
      indent: 2em,
      spacing: 0.65em,
      marker: "-",
    )

    set enum(
      indent: 2em,
      spacing: 0.65em,
    )

    set heading(numbering: "1.")

    show heading.where(level: 1): h => {
      set align(center)
      set text(
        weight: "bold",
        size: 12pt,
      )

      pagebreak(weak: true)
      if h.numbering != none [
        #counter(heading).display(h.numbering) #h.body
      ] else [
        #h.body
      ]
    }

    show heading.where(level: 2): h => {
      set text(
        weight: "bold",
        size: 12pt,
      )

      block(inset: (left: 1em))[#counter(heading).display() #h.body]
    }

    show heading.where(level: 3): h => {
      set text(weight: "bold", size: 12pt)

      block(inset: (left: 3em))[#counter(heading).display() #h.body]
    }

    pagebreak(weak: true)
    align(center, text(weight: "bold", size: 12pt, [АННОТАЦИЯ]))
    document-annotation(id)

    outline-block

    body
  }

  set text(
    lang: "ru",
    size: 12pt,
    font: "Times New Roman",
  )

  approval-page
  title-page
  normal-pages
  list-registration-changes
}
