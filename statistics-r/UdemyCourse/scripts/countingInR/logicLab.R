has_cap = T
has_receipt = F
paid_by_card = F
has_code = T
code_winning = F
is_company_related = F

has_cap & (has_receipt | paid_by_card) & has_code & code_winning & !is_company_related


p <- T
q <- F

(!p & !q ) == !(p | q)
