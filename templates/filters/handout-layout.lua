-- Keep headings and short pedagogical notes from being stranded at page bottoms.
-- This filter changes layout only; it does not change pedagogical content.
local function latex_needspace(lines)
  return pandoc.RawBlock('latex', string.format('\\Needspace{%d\\baselineskip}', lines))
end

function Header(el)
  if not FORMAT:match('latex') then return nil end
  local n = 4
  if el.level == 1 then n = 6
  elseif el.level == 2 then n = 5
  elseif el.level == 3 then n = 4 end
  return {latex_needspace(n), el}
end

function BlockQuote(el)
  if not FORMAT:match('latex') then return nil end
  return {latex_needspace(4), el}
end
