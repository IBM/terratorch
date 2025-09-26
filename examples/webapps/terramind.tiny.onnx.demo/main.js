import './style.scss';
import '@carbon/web-components/es/components/button/button.js';
import '@carbon/web-components/es/components/ui-shell/index';
import '@carbon/web-components/es/components/checkbox/index';
import '@carbon/web-components/es/components/content-switcher/index';
import '@carbon/web-components/es/components/skip-to-content/index.js';
import '@carbon/web-components/es/components/breadcrumb/index';
import '@carbon/web-components/es/components/tabs/index';

const bodyEl = document.querySelector('body');

const handleGlobalActionClick = (ev) => {
  const targetPanelId = ev.currentTarget.getAttribute('panel-id');
  const panels = document.querySelectorAll('cds-header-panel');
  // check to see if other panels are open and close them
  panels.forEach((panel) => {
    if (panel.id !== targetPanelId) {
      panel.expanded = false;
    }
  });
};
const globalActions = document.querySelectorAll('cds-header-global-action');
[...globalActions].forEach((action) =>
  action.addEventListener('click', handleGlobalActionClick),
);

const handleSwitch = (ev) => {
  // Applies new theme or defers to system preferences by removing theme
  switch (ev.detail.item.value) {
    case 'light':
      bodyEl.classList.remove('g100');
      bodyEl.classList.add('g10');
      break;
    case 'dark':
      bodyEl.classList.remove('g10');
      bodyEl.classList.add('g100');
      break;
    default:
      bodyEl.classList.remove('g10');
      bodyEl.classList.remove('g100');
  }
};
document
  .querySelector('.theme-selector')
  .addEventListener('cds-content-switcher-selected', handleSwitch);

const handleHeaderCompliment = (ev) => {
  document
    .querySelector('cds-header')
    .classList.toggle('compliment', ev.target.checked);
};
document
  .querySelector('.theme-header__compliment')
  .addEventListener('cds-checkbox-changed', handleHeaderCompliment);
